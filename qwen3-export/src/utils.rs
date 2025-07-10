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

    pub fn set_current(&self, current: usize) {
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
                "\r{}: [{}] {}/{} ({}%)",
                self.label, bar, current, self.total, percent
            );
            io::stdout().flush().unwrap_or(());

            if current >= self.total {
                println!(); // New line when complete
            }
        }
    }
}
