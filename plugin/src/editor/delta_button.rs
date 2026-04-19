// Spectral Compressor: an FFT based compressor
// Copyright (C) 2021-2024 Robbert van der Helm
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use nih_plug_vizia::vizia::prelude::*;

/// A toggle button for the delta-listen feature. When active the plugin outputs the difference
/// between the input and the processed signal so the user can hear what the compressor is doing.
pub struct DeltaButton {
    active: Arc<AtomicBool>,
}

impl DeltaButton {
    pub fn new<L, T>(cx: &mut Context, lens: L, label: impl Res<T> + Clone) -> Handle<'_, Self>
    where
        L: Lens<Target = Arc<AtomicBool>>,
        T: ToString,
    {
        Self {
            active: lens.get(cx),
        }
        .build(cx, |cx| {
            Label::new(cx, label).hoverable(false);
        })
        .checked(lens.map(|v| v.load(Ordering::Relaxed)))
        // Reuse param-button styling
        .class("editor-mode")
    }
}

impl View for DeltaButton {
    fn element(&self) -> Option<&'static str> {
        // Reuse the styling from param-button
        Some("param-button")
    }

    /// Toggles delta-listen mode on left click. Writes straight to the shared atomic
    /// so the DSP side picks it up on the next process call.
    fn event(&mut self, _cx: &mut EventContext, event: &mut Event) {
        event.map(|window_event, meta| match window_event {
            WindowEvent::MouseDown(MouseButton::Left)
            | WindowEvent::MouseDoubleClick(MouseButton::Left)
            | WindowEvent::MouseTripleClick(MouseButton::Left) => {
                let current = self.active.load(Ordering::Relaxed);
                self.active.store(!current, Ordering::Relaxed);
                meta.consume();
            }
            _ => {}
        });
    }
}
