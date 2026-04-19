use nih_plug::prelude::*;

use polarity_sc_dark::SpectralCompressor;

fn main() {
    // The deeply nested GUI widget tree can overflow the default stack size in
    // debug builds, so we spawn the standalone on a thread with a larger stack.
    let builder = std::thread::Builder::new()
        .name("main".to_string())
        .stack_size(8 * 1024 * 1024); // 8 MB
    let handler = builder
        .spawn(|| {
            nih_export_standalone::<SpectralCompressor>();
        })
        .expect("Failed to spawn main thread");
    handler.join().unwrap();
}
