use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

use atomic_refcell::AtomicRefCell;
use nih_plug_vizia::vizia::prelude::*;

use crate::frozen_ir::{write_frozen_ir_wav, FrozenIrData};

/// Exports the current frozen compressor snapshot as a WAV IR file.
pub struct ExportIrButton {
    frozen_ir_data: Arc<AtomicRefCell<triple_buffer::Output<FrozenIrData>>>,
}

impl ExportIrButton {
    pub fn new<LData, T>(
        cx: &mut Context,
        frozen_ir_lens: LData,
        label: impl Res<T> + Clone,
    ) -> Handle<'_, Self>
    where
        LData: Lens<Target = Arc<AtomicRefCell<triple_buffer::Output<FrozenIrData>>>>,
        T: ToString,
    {
        Self {
            frozen_ir_data: frozen_ir_lens.get(cx),
        }
        .build(cx, |cx| {
            Label::new(cx, label).hoverable(false);
        })
        .class("editor-mode")
    }
}

impl View for ExportIrButton {
    fn element(&self) -> Option<&'static str> {
        Some("param-button")
    }

    /// Starts the export when the button is clicked. The file dialog and the actual
    /// WAV write happen on a worker thread so the editor stays responsive.
    fn event(&mut self, cx: &mut EventContext, event: &mut Event) {
        event.map(|window_event, meta| match window_event {
            WindowEvent::MouseDown(MouseButton::Left)
            | WindowEvent::MouseDoubleClick(MouseButton::Left)
            | WindowEvent::MouseTripleClick(MouseButton::Left) => {
                meta.consume();
                if cx.is_disabled() {
                    return;
                }

                start_export(self.frozen_ir_data.clone());
            }
            _ => {}
        });
    }
}

/// Grabs the latest valid frozen snapshot and kicks off the async export.
fn start_export(frozen_ir_data: Arc<AtomicRefCell<triple_buffer::Output<FrozenIrData>>>) {
    let snapshot = if let Ok(mut frozen_ir_data) = frozen_ir_data.try_borrow_mut() {
        frozen_ir_data.read().clone()
    } else {
        return;
    };
    if !snapshot.valid {
        return;
    }

    thread::Builder::new()
        .name(String::from("polarity-ir-export"))
        .spawn(move || {
            let _ = export_current_snapshot(snapshot);
        })
        .expect("failed to spawn IR export thread");
}

/// Opens a save dialog and writes the passed frozen snapshot to a WAV file. Returns
/// the chosen path on success, `None` when the user cancels, or an error message on
/// failure.
fn export_current_snapshot(snapshot: FrozenIrData) -> Result<Option<PathBuf>, String> {
    let Some(path) = rfd::FileDialog::new()
        .add_filter("Wave", &["wav"])
        .set_file_name("polarity-sc-dark-freeze-ir.wav")
        .save_file()
    else {
        return Ok(None);
    };

    write_frozen_ir_wav(&snapshot, &path)?;
    Ok(Some(path))
}
