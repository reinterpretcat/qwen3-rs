use anyhow::{Context, Result};
use memmap2::Mmap;
use std::fs::File;
use std::slice;

#[derive(Debug)]
pub(crate) struct MemoryMapper {
    mmap: Mmap,
    offset: usize,
}

impl MemoryMapper {
    pub fn new(file: File) -> Result<Self> {
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file).context("Failed to create memory mapping")? };
        Ok(Self { mmap, offset: 0 })
    }

    pub fn get_f32_slice(&mut self, count: usize) -> Result<&[f32]> {
        let bytes_needed = count * std::mem::size_of::<f32>();

        if self.offset + bytes_needed > self.mmap.len() {
            anyhow::bail!(
                "Insufficient data: need {} bytes, have {} remaining",
                bytes_needed,
                self.mmap.len() - self.offset
            );
        }

        let byte_slice = &self.mmap[self.offset..self.offset + bytes_needed];
        self.offset += bytes_needed;

        // SAFETY: We're casting from &[u8] to &[f32]
        // This is safe because:
        // 1. We've verified the slice has the correct length
        // 2. f32 has less strict alignment requirements than most types
        // 3. The checkpoint file is assumed to be correctly formatted
        let f32_slice = unsafe { slice::from_raw_parts(byte_slice.as_ptr() as *const f32, count) };

        Ok(f32_slice)
    }

    pub fn get_bytes(&mut self, count: usize) -> Result<&[u8]> {
        if self.offset + count > self.mmap.len() {
            anyhow::bail!("Insufficient data: need {} bytes, have {} remaining", count, self.mmap.len() - self.offset);
        }

        let result = &self.mmap[self.offset..self.offset + count];
        self.offset += count;
        Ok(result)
    }

    pub fn skip(&mut self, bytes: usize) -> Result<()> {
        if self.offset + bytes > self.mmap.len() {
            anyhow::bail!("Cannot skip {} bytes: insufficient data", bytes);
        }
        self.offset += bytes;
        Ok(())
    }
}
