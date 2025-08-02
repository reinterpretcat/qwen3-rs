use anyhow::{Context, Result};
use log::info;
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::{
    collections::{HashMap, VecDeque},
    fs::File,
    mem,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

/// Memory-efficient tensor reader from SafeTensors files
#[derive(Debug)]
pub(crate) struct TensorReader {
    safetensors_files: Vec<PathBuf>,   // Just store file paths, not data
    mmap_cache: Arc<Mutex<MmapCache>>, // LRU cache for memory mappings
}

impl TensorReader {
    pub fn new(model_path: &Path) -> Result<Self> {
        let safetensors_files = std::fs::read_dir(model_path)
            .with_context(|| format!("Failed to read directory: {}", model_path.display()))?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();

                // Check if it's a .safetensors file
                matches!(path.extension(), Some(ext) if ext == "safetensors").then_some(path)
            })
            .collect::<Vec<_>>();

        if safetensors_files.is_empty() {
            anyhow::bail!("No SafeTensors files found in {}", model_path.display());
        }

        info!("Found {} safetensor files", safetensors_files.len());

        Ok(TensorReader {
            safetensors_files,
            mmap_cache: Arc::new(Mutex::new(MmapCache::new(10))), // Max 10 cached files
        })
    }

    /// Load a specific tensor by name, converting from BF16/F32 to F32
    pub fn load_tensor(&self, tensor_name: &str) -> Result<Option<Vec<f32>>> {
        self.safetensors_files
            .iter()
            .find_map(|filename| {
                // Use cached memory mapping
                // TODO we deserialize here each time, might be not so efficient
                let mmap = self.get_mmap(filename).ok()?;
                let safetensors = SafeTensors::deserialize(&mmap)
                    .with_context(|| format!("Failed to deserialize {}", filename.display()))
                    .ok()?;

                // Try to find the tensor in this file
                safetensors
                    .tensor(tensor_name)
                    .ok()
                    .and_then(|tensor_view| {
                        Self::convert_tensor_to_f32(&tensor_view, tensor_name).ok()
                    })
            })
            .map_or(Ok(None), |data| Ok(Some(data)))
    }

    /// Read all tensor files and lists all available tensor names in the model.
    #[cfg(debug_assertions)]
    pub fn list_tensor_names(&self) -> Result<HashMap<String, String>> {
        let mut all_tensor_names = HashMap::new();

        for filename in &self.safetensors_files {
            let mmap = self.get_mmap(filename)?;
            let safetensors = SafeTensors::deserialize(&mmap)
                .with_context(|| format!("Failed to deserialize {}", filename.display()))?;

            all_tensor_names.insert(
                filename.to_string_lossy().into_owned(),
                safetensors
                    .names()
                    .iter()
                    .map(|&name| name.clone())
                    .collect(),
            );
        }

        Ok(all_tensor_names)
    }

    /// Convert tensor data to f32 based on its data type
    fn convert_tensor_to_f32(
        tensor_view: &safetensors::tensor::TensorView,
        tensor_name: &str,
    ) -> Result<Vec<f32>> {
        let tensor_data = tensor_view.data();
        let shape = tensor_view.shape();
        let dtype = tensor_view.dtype();
        let expected_elements = shape.iter().product::<usize>();

        match dtype {
            safetensors::Dtype::F32 => {
                Self::validate_tensor_size(
                    tensor_data.len(),
                    expected_elements * mem::size_of::<f32>(),
                    tensor_name,
                    "F32",
                )?;
                Ok(Self::convert_f32_data(tensor_data))
            }
            safetensors::Dtype::BF16 => {
                Self::validate_tensor_size(
                    tensor_data.len(),
                    expected_elements * 2,
                    tensor_name,
                    "BF16",
                )?;
                Ok(Self::convert_bf16_data(tensor_data))
            }
            _ => anyhow::bail!("Unsupported tensor dtype {:?} for {}", dtype, tensor_name),
        }
    }

    /// Validate tensor data size matches expected size
    fn validate_tensor_size(
        actual_bytes: usize,
        expected_bytes: usize,
        tensor_name: &str,
        dtype_name: &str,
    ) -> Result<()> {
        if actual_bytes != expected_bytes {
            anyhow::bail!(
                "{} tensor {} size mismatch. Expected {} bytes, got {}",
                dtype_name,
                tensor_name,
                expected_bytes,
                actual_bytes
            );
        }
        Ok(())
    }

    /// Convert F32 tensor data
    fn convert_f32_data(data: &[u8]) -> Vec<f32> {
        data.chunks_exact(mem::size_of::<f32>())
            .map(|chunk| {
                let bytes: [u8; 4] = chunk.try_into().expect("chunk size is guaranteed to be 4");
                f32::from_le_bytes(bytes)
            })
            .collect()
    }

    /// Convert BF16 tensor data to F32
    fn convert_bf16_data(data: &[u8]) -> Vec<f32> {
        data.chunks_exact(2)
            .map(|chunk| {
                let [low, high] = chunk else {
                    unreachable!("chunks_exact(2) guarantees 2 bytes")
                };
                // BF16 to F32: BF16 is the upper 16 bits of F32
                let bf16_bits = u16::from_le_bytes([*low, *high]);
                let f32_bits = (bf16_bits as u32) << 16;
                f32::from_bits(f32_bits)
            })
            .collect()
    }

    /// Get or create a cached memory mapping for a file
    fn get_mmap(&self, path: &Path) -> Result<Arc<Mmap>> {
        let mut cache = self
            .mmap_cache
            .lock()
            .map_err(|_| anyhow::anyhow!("Failed to acquire cache lock"))?;

        if let Some(cached_mmap) = cache.get(path) {
            return Ok(cached_mmap);
        }

        // Create new mapping
        let file =
            File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;

        // SAFETY: All file-backed memory map constructors are marked `unsafe` because of the potential for
        // *Undefined Behavior* (UB) using the map if the underlying file is subsequently modified, in or
        // out of process.
        let mmap = Arc::new(
            unsafe { Mmap::map(&file) }
                .with_context(|| format!("Failed to memory map {}", path.display()))?,
        );

        // Cache it with LRU eviction
        cache.insert(path.to_path_buf(), Arc::clone(&mmap));
        Ok(mmap)
    }

    /// Clear the memory mapping cache to free memory
    #[allow(dead_code)]
    pub fn clear_cache(&self) -> Result<()> {
        let mut cache = self
            .mmap_cache
            .lock()
            .map_err(|_| anyhow::anyhow!("Failed to acquire cache lock"))?;
        cache.clear();
        Ok(())
    }
}

/// Memory-efficient tensor reader with LRU cache
#[derive(Debug)]
struct MmapCache {
    cache: HashMap<PathBuf, Arc<Mmap>>,
    access_order: VecDeque<PathBuf>,
    max_size: usize,
}

impl MmapCache {
    fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            access_order: VecDeque::new(),
            max_size,
        }
    }

    fn get(&mut self, path: &Path) -> Option<Arc<Mmap>> {
        if let Some(mmap) = self.cache.get(path) {
            // Move to front (most recently used)
            if let Some(pos) = self.access_order.iter().position(|p| p == path) {
                self.access_order.remove(pos);
            }
            self.access_order.push_front(path.to_path_buf());
            Some(Arc::clone(mmap))
        } else {
            None
        }
    }

    fn insert(&mut self, path: PathBuf, mmap: Arc<Mmap>) {
        // Remove if already exists
        if self.cache.contains_key(&path) {
            if let Some(pos) = self.access_order.iter().position(|p| p == &path) {
                self.access_order.remove(pos);
            }
        }

        // Evict least recently used if cache is full
        while self.cache.len() >= self.max_size {
            if let Some(lru_path) = self.access_order.pop_back() {
                self.cache.remove(&lru_path);
            } else {
                break;
            }
        }

        // Insert new mapping
        self.cache.insert(path.clone(), mmap);
        self.access_order.push_front(path);
    }

    fn clear(&mut self) {
        self.cache.clear();
        self.access_order.clear();
    }
}
