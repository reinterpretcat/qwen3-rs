use std::{
    io::{self, Write},
    sync::atomic::{AtomicUsize, Ordering},
};

/// Progress tracker for showing progress bars
#[derive(Debug)]
pub(crate) struct ProgressTracker {
    total: usize,
    completed: AtomicUsize,
    last_displayed: AtomicUsize,
    label: String,
}

impl ProgressTracker {
    pub fn new(total: usize, label: &str) -> Self {
        Self {
            total,
            completed: AtomicUsize::new(0),
            last_displayed: AtomicUsize::new(0),
            label: label.to_string(),
        }
    }

    pub fn set_current(&self, current: usize, description: Option<&str>) {
        self.completed.store(current, Ordering::Relaxed);
        let percent = (current * 100) / self.total;
        let last_displayed = self.last_displayed.load(Ordering::Relaxed);
        let last_percent = (last_displayed * 100) / self.total;

        // Update display every 1% or on key milestones
        if current == 0
            || percent > last_percent
            || current >= self.total
            || current - last_displayed >= 10
        {
            self.last_displayed.store(current, Ordering::Relaxed);
            let bar_width = 30;
            let filled = (current * bar_width) / self.total;
            let bar = "█".repeat(filled) + &"░".repeat(bar_width - filled);

            print!(
                "\r{}: [{bar}] {current}/{} ({percent}%): {}",
                self.label,
                self.total,
                fixed_len(description.unwrap_or_default(), 42)
            );
            io::stdout().flush().unwrap_or(());

            if current >= self.total {
                println!(); // New line when complete
            }
        }
    }
}

fn fixed_len(description: &str, width: usize) -> String {
    let mut desc = description.to_string();
    if desc.len() > width {
        // Cut and add ".."
        desc.truncate(width.saturating_sub(2));
        desc.push_str("..");
    } else if desc.len() < width {
        // Pad with spaces
        desc = format!("{:width$}", desc, width = width);
    }
    desc
}

#[cfg(test)]
mod tests {

    use rayon::prelude::*;

    #[test]
    fn test_par_chunks_zip() {
        let mut data1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let mut data2 = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let n_heads = 4;

        data1
            .par_chunks_mut(2)
            .zip(data2.par_chunks_mut(2))
            .zip((0..n_heads).into_par_iter())
            .for_each(|((chunk1, chunk2), head_idx)| {
                for (a, b) in chunk1.iter_mut().zip(chunk2.iter_mut()) {
                    *a += head_idx;
                    *b += head_idx * 10;
                }
            });

        assert_eq!(data1, vec![1, 2, 4, 5, 7, 8, 10, 11]);
        assert_eq!(data2, vec![10, 20, 40, 50, 70, 80, 100, 110]);
    }

    #[test]
    fn test_attention_pattern() {
        struct MockState {
            att: Vec<f32>,
            xb: Vec<f32>,
        }

        let seq_len = 4;
        let head_dim = 8;
        let n_heads = 4;

        let mut state = MockState {
            att: vec![0.0; n_heads * seq_len],
            xb: vec![0.0; n_heads * head_dim],
        };

        state
            .att
            .par_chunks_mut(seq_len)
            .zip(state.xb.par_chunks_mut(head_dim))
            .zip((0..n_heads).into_par_iter())
            .for_each(|((att_slice, xb_slice), head_idx)| {
                for val in att_slice.iter_mut() {
                    *val = head_idx as f32;
                }
                for val in xb_slice.iter_mut() {
                    *val = (head_idx * 10) as f32;
                }
            });

        for head in 0..n_heads {
            for i in 0..seq_len {
                assert_eq!(state.att[head * seq_len + i], head as f32);
            }
            for i in 0..head_dim {
                assert_eq!(state.xb[head * head_dim + i], (head * 10) as f32);
            }
        }
    }
}
