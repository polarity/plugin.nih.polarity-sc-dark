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

use atomic_float::AtomicF32;
use atomic_refcell::AtomicRefCell;
use nih_plug::prelude::*;
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::widgets::*;
use nih_plug_vizia::{assets, create_vizia_editor, ViziaState, ViziaTheming};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

#[cfg(feature = "hot-reload")]
use std::path::PathBuf;

use self::analyzer::Analyzer;
use self::delta_button::DeltaButton;
use self::export_ir_button::ExportIrButton;
use self::match_button::MatchButton;
use crate::analyzer::AnalyzerData;
use crate::compressor_bank::ThresholdMode;
use crate::frozen_ir::FrozenIrData;
use crate::match_level::{MatchResult, MatchRuntime};
use crate::{SpectralCompressor, SpectralCompressorParams};

mod analyzer;
mod delta_button;
mod export_ir_button;
mod match_button;
pub mod theme;

/// The entire GUI's width, in logical pixels.
const GUI_WIDTH: u32 = 1360;
/// The entire GUI's height, in logical pixels.
const GUI_HEIGHT: u32 = 530;

// All GUI colors are defined in `editor::theme`.

#[derive(Clone, Lens)]
/// Editor state passed into the vizia views. This bundles the parameter tree and the
/// shared runtime handles the custom widgets need to talk to the audio thread.
pub(crate) struct Data {
    pub(crate) params: Arc<SpectralCompressorParams>,
    /// Display labels for the threshold mode pick list.
    pub(crate) mode_options: Vec<String>,

    /// When true the plugin outputs the delta signal (input minus processed).
    pub(crate) delta_active: Arc<AtomicBool>,
    /// Request/result handoff for the editor-triggered output gain match.
    pub(crate) match_runtime: Arc<MatchRuntime>,

    pub(crate) analyzer_data: Arc<AtomicRefCell<triple_buffer::Output<AnalyzerData>>>,
    pub(crate) frozen_ir_data: Arc<AtomicRefCell<triple_buffer::Output<FrozenIrData>>>,
    /// Used by the analyzer to determine which FFT bins belong to which frequencies.
    pub(crate) sample_rate: Arc<AtomicF32>,
}

impl Model for Data {}

#[derive(Lens)]
/// Editor-local state for the status label and the export button's enabled flag.
struct ExportUiModel {
    can_export: bool,
    status_text: String,
    frozen_ir_data: Arc<AtomicRefCell<triple_buffer::Output<FrozenIrData>>>,
    params: Arc<SpectralCompressorParams>,
    gui_context: Arc<dyn GuiContext>,
}

impl ExportUiModel {
    fn build(
        cx: &mut Context,
        frozen_ir_data: Arc<AtomicRefCell<triple_buffer::Output<FrozenIrData>>>,
        params: Arc<SpectralCompressorParams>,
        gui_context: Arc<dyn GuiContext>,
    ) {
        let can_export = Self::read_can_export(&frozen_ir_data);
        Self {
            can_export,
            status_text: String::new(),
            frozen_ir_data,
            params,
            gui_context,
        }
        .build(cx);
    }

    fn read_can_export(
        frozen_ir_data: &Arc<AtomicRefCell<triple_buffer::Output<FrozenIrData>>>,
    ) -> bool {
        if let Ok(mut frozen_ir_data) = frozen_ir_data.try_borrow_mut() {
            frozen_ir_data.read().valid
        } else {
            false
        }
    }

    fn refresh_export_availability(&mut self) {
        self.can_export = Self::read_can_export(&self.frozen_ir_data);
    }
}

impl Model for ExportUiModel {
    fn event(&mut self, cx: &mut EventContext, event: &mut Event) {
        // Events from the match poll thread arrive here as `MatchEvent::Result`.
        event.map(|match_event: &match_button::MatchEvent, _meta| match match_event {
            match_button::MatchEvent::Started => {
                self.status_text = String::from("Matching...");
            }
            match_button::MatchEvent::Result(result) => {
                match result {
                    MatchResult::Matched(output_gain) => {
                        let output_gain = *output_gain;
                        let setter = ParamSetter::new(self.gui_context.as_ref());
                        setter.begin_set_parameter(&self.params.global.output_gain);
                        setter.set_parameter(&self.params.global.output_gain, output_gain);
                        setter.end_set_parameter(&self.params.global.output_gain);
                        cx.emit(RawParamEvent::ParametersChanged);

                        let output_gain_db = nih_plug::prelude::util::gain_to_db(output_gain);
                        self.status_text = format!("Matched {output_gain_db:+.2} dB");
                    }
                    MatchResult::Failed => {
                        self.status_text = String::from("Match failed: signal too quiet.");
                    }
                }

                self.refresh_export_availability();
            }
        });

        // Keep export state in sync when parameters are changed from the UI/host.
        event.map(|_param_event: &RawParamEvent, _meta| {
            self.refresh_export_availability();
        });
    }
}

/// Creates the default persisted editor window state.
pub(crate) fn default_state() -> Arc<ViziaState> {
    ViziaState::new(move || (GUI_WIDTH, GUI_HEIGHT))
}

pub(crate) fn create(editor_state: Arc<ViziaState>, editor_data: Data) -> Option<Box<dyn Editor>> {
    create_vizia_editor(
        editor_state,
        ViziaTheming::Custom,
        move |cx, gui_context| {
            assets::register_noto_sans_light(cx);
            assets::register_noto_sans_thin(cx);

            // In hot-reload mode the CSS is read from disk so you can tweak it without
            // recompiling.  Close and reopen the editor window (or the standalone) to
            // pick up changes.
            #[cfg(feature = "hot-reload")]
            {
                let css_path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "src", "editor", "theme.css"]
                    .iter()
                    .collect();
                if let Err(err) = cx.add_stylesheet(css_path.clone()) {
                    nih_error!("[hot-reload] Could not load {}: {err:?}", css_path.display());
                } else {
                    nih_log!("[hot-reload] Loaded CSS from {}", css_path.display());
                }
            }
            #[cfg(not(feature = "hot-reload"))]
            if let Err(err) = cx.add_stylesheet(include_style!("src/editor/theme.css")) {
                nih_error!("Failed to load stylesheet: {err:?}")
            }

            editor_data.clone().build(cx);
            ExportUiModel::build(
                cx,
                editor_data.frozen_ir_data.clone(),
                editor_data.params.clone(),
                gui_context,
            );

            HStack::new(cx, |cx| {
                main_column(cx);
                analyzer_column(cx);
            });
        },
    )
}

/// Builds the left/main side of the editor.
fn main_column(cx: &mut Context) {
    VStack::new(cx, |cx| {
        header_bar(cx);
        global_and_threshold_row(cx);
        compressor_columns(cx);
    })
    .class("main-column");
}

/// Header row containing Delta/Match/Bypass controls, status, and title/version.
fn header_bar(cx: &mut Context) {
    let global_params = Data::params.map(|p| p.global.clone());

    HStack::new(cx, |cx| {
        HStack::new(cx, |cx| {
            DeltaButton::new(cx, Data::delta_active, "Delta").class("header-btn");
            MatchButton::new(
                cx,
                Data::match_runtime,
                Data::delta_active,
                global_params.clone(),
                "Match",
            )
            .class("header-btn");
            ParamButton::new(cx, global_params, |params| &params.bypass)
                .with_label("Bypass")
                .class("header-btn");
        })
        .class("header-button-group");

        Label::new(cx, ExportUiModel::status_text).class("match-status");

        HStack::new(cx, |cx| {
            Label::new(cx, "Polarity-SC")
                .class("plugin-title")
                .on_mouse_down(|_, _| {
                    // `CARGO_PKG_HOMEPAGE` is `""` when no `homepage` is set in
                    // Cargo.toml, and `open::that("")` opens Explorer in the
                    // current working directory on Windows.
                    let url = SpectralCompressor::URL;
                    if url.is_empty() {
                        return;
                    }
                    // Spawn on a background thread: on Windows, `open::that`
                    // ultimately calls `ShellExecuteW`, which can hang or crash
                    // when invoked from a plugin host's UI thread (COM apartment
                    // mismatches, reentrant message pumping, etc.).
                    std::thread::spawn(move || {
                        let result = open::that(url);
                        if cfg!(debug_assertions) && result.is_err() {
                            nih_debug_assert_failure!(
                                "Failed to open web browser: {:?}",
                                result
                            );
                        }
                    });
                });
            Label::new(cx, SpectralCompressor::VERSION).class("version-label");
        })
        .class("title-group");
    })
    .class("header-bar");
}

/// First content row with global controls and threshold controls.
fn global_and_threshold_row(cx: &mut Context) {
    HStack::new(cx, |cx| {
        make_column(cx, "Globals", None, |cx| {
            global_column(cx);
        });

        make_column(cx, "Threshold", None, |cx| {
            threshold_column(cx);
        });
    })
    .size(Auto);
}

/// Renders the global parameter section with custom row ordering/grouping.
fn global_column(cx: &mut Context) {
    let global_params = Data::params.map(|p| p.global.clone());

    GenericUi::new_custom(cx, global_params, |cx, param_ptr| {
        let name = unsafe { param_ptr.name() };
        if name == "Output Gain"
            || name == "Mix"
            || name == "Window Size"
            || name == "Window Overlap"
            || name == "Attack"
            || name == "Release"
        {
            HStack::new(cx, |cx| {
                Label::new(cx, name).class("label");
                GenericUi::draw_widget(cx, global_params, param_ptr);
            })
            .class("row");
        } else if name == "Freeze" {
            HStack::new(cx, |cx| {
                Label::new(cx, name).class("label");
                HStack::new(cx, |cx| {
                    ParamButton::new(cx, global_params, |params| &params.compressor_freeze)
                        .with_label("Freeze")
                        .class("freeze-button");
                    ExportIrButton::new(cx, Data::frozen_ir_data, "Export IR...")
                        .disabled(ExportUiModel::can_export.map(|can_export| !*can_export))
                        .class("freeze-button");
                })
                .class("freeze-button-group");
            })
            .class("row");
        }
    });
}

/// Renders threshold parameters.
fn threshold_column(cx: &mut Context) {
    let threshold_params = Data::params.map(|p| p.threshold.clone());

    GenericUi::new_custom(cx, threshold_params, |cx, param_ptr| {
        let name = unsafe { param_ptr.name() };

        HStack::new(cx, |cx| {
            Label::new(cx, name).class("label");

            if name == "Mode" {
                mode_picklist(cx);
            } else {
                GenericUi::draw_widget(cx, threshold_params, param_ptr);
            }
        })
        .class("row");
    });
}

/// Builds side-by-side upwards and downwards compressor columns.
fn compressor_columns(cx: &mut Context) {
    HStack::new(cx, |cx| {
        upwards_column(cx);
        downwards_column(cx);
    })
    .size(Auto);
}

/// Renders controls for the upwards compressor bank.
fn upwards_column(cx: &mut Context) {
    make_column(cx, "Upwards", Some("upwards-col"), |cx| {
        let upwards_compressor_params = Data::params.map(|p| p.compressors.upwards.clone());
        GenericUi::new_custom(cx, upwards_compressor_params, |cx, param_ptr| {
            HStack::new(cx, |cx| {
                Label::new(
                    cx,
                    unsafe { param_ptr.name() }
                        .strip_prefix("Upwards ")
                        .expect("missing prefix"),
                )
                .class("label");

                GenericUi::draw_widget(cx, upwards_compressor_params, param_ptr);
            })
            .class("row");
        });
    });
}

/// Renders controls for the downwards compressor bank.
fn downwards_column(cx: &mut Context) {
    make_column(cx, "Downwards", Some("downwards-col"), |cx| {
        let downwards_compressor_params = Data::params.map(|p| p.compressors.downwards.clone());
        GenericUi::new_custom(cx, downwards_compressor_params, |cx, param_ptr| {
            HStack::new(cx, |cx| {
                Label::new(
                    cx,
                    unsafe { param_ptr.name() }
                        .strip_prefix("Downwards ")
                        .expect("missing prefix"),
                )
                .class("label");

                GenericUi::draw_widget(cx, downwards_compressor_params, param_ptr);
            })
            .class("row");
        });
    });
}

/// Creates the analyzer panel on the right side of the editor.
fn analyzer_column(cx: &mut Context) {
    Analyzer::new(cx, Data::analyzer_data, Data::sample_rate, Data::params);
}

/// Custom threshold mode selector that writes raw parameter automation events.
fn mode_picklist(cx: &mut Context) {
    let params = Data::params.get(cx);

    PickList::new(
        cx,
        Data::mode_options,
        Data::params.map(|p| p.threshold.mode.value().to_index()),
        true,
    )
    .class("mode-picklist")
    .on_select(move |cx, index| {
        if params.threshold.mode.value().to_index() == index {
            return;
        }

        let new_mode = ThresholdMode::from_index(index);
        let mode_param = params.threshold.mode.as_ptr();
        let normalized = params.threshold.mode.preview_normalized(new_mode);

        cx.emit(RawParamEvent::BeginSetParameter(mode_param));
        cx.emit(RawParamEvent::SetParameterNormalized(
            mode_param, normalized,
        ));
        cx.emit(RawParamEvent::EndSetParameter(mode_param));
    });
}

/// Utility helper to render a titled editor column with optional CSS class.
fn make_column(
    cx: &mut Context,
    title: &str,
    css_class: Option<&str>,
    contents: impl FnOnce(&mut Context),
) {
    let col = VStack::new(cx, |cx| {
        Label::new(cx, title).class("column-title");

        contents(cx);
    })
    .class("column");

    if let Some(cls) = css_class {
        col.class(cls);
    }
}
