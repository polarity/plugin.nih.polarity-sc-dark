use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::widgets::param_base::ParamWidgetBase;

use crate::match_curve::{MatchCurveResult, MatchCurveRuntime};
use crate::GlobalParams;

#[derive(Debug)]
pub enum MatchCurveEvent {
    Started,
    Result(MatchCurveResult),
    ClearRequested,
}

pub struct MatchCurveButton {
    match_curve_runtime: Arc<MatchCurveRuntime>,
    delta_active: Arc<AtomicBool>,
    bypass_param: ParamWidgetBase,
}

pub struct ClearCurveButton;

impl ClearCurveButton {
    pub fn new<T>(cx: &mut Context, label: impl Res<T> + Clone) -> Handle<'_, Self>
    where
        T: ToString,
    {
        Self.build(cx, |cx| {
            Label::new(cx, label).hoverable(false);
        })
        .class("editor-mode")
    }
}

impl View for ClearCurveButton {
    fn element(&self) -> Option<&'static str> {
        Some("param-button")
    }

    fn event(&mut self, cx: &mut EventContext, event: &mut Event) {
        event.map(|window_event, meta| match window_event {
            WindowEvent::MouseDown(MouseButton::Left)
            | WindowEvent::MouseDoubleClick(MouseButton::Left)
            | WindowEvent::MouseTripleClick(MouseButton::Left) => {
                meta.consume();
                cx.emit(MatchCurveEvent::ClearRequested);
            }
            _ => {}
        });
    }
}

impl MatchCurveButton {
    pub fn new<LMatch, LDelta, LGlobal, T>(
        cx: &mut Context,
        match_curve_lens: LMatch,
        delta_lens: LDelta,
        global_params: LGlobal,
        label: impl Res<T> + Clone,
    ) -> Handle<'_, Self>
    where
        LMatch: Lens<Target = Arc<MatchCurveRuntime>>,
        LDelta: Lens<Target = Arc<AtomicBool>>,
        LGlobal: Lens<Target = Arc<GlobalParams>> + Clone,
        T: ToString,
    {
        Self {
            match_curve_runtime: match_curve_lens.get(cx),
            delta_active: delta_lens.get(cx),
            bypass_param: ParamWidgetBase::new(cx, global_params, |params| &params.bypass),
        }
        .build(cx, |cx| {
            Label::new(cx, label).hoverable(false);
        })
        .class("editor-mode")
    }
}

impl View for MatchCurveButton {
    fn element(&self) -> Option<&'static str> {
        Some("param-button")
    }

    fn event(&mut self, cx: &mut EventContext, event: &mut Event) {
        event.map(|window_event, meta| match window_event {
            WindowEvent::MouseDown(MouseButton::Left)
            | WindowEvent::MouseDoubleClick(MouseButton::Left)
            | WindowEvent::MouseTripleClick(MouseButton::Left) => {
                meta.consume();

                self.delta_active.store(false, Ordering::Relaxed);
                if self.bypass_param.unmodulated_normalized_value() >= 0.5 {
                    self.bypass_param.begin_set_parameter(cx);
                    self.bypass_param.set_normalized_value(cx, 0.0);
                    self.bypass_param.end_set_parameter(cx);
                }
                cx.emit(MatchCurveEvent::Started);
                self.match_curve_runtime.request();

                let runtime = self.match_curve_runtime.clone();
                cx.spawn(move |proxy| {
                    let deadline = Instant::now() + Duration::from_secs(15);
                    loop {
                        if let Some(result) = runtime.take_finished_result() {
                            let _ = proxy.emit(MatchCurveEvent::Result(result));
                            return;
                        }
                        if Instant::now() >= deadline {
                            let _ = proxy.emit(MatchCurveEvent::Result(MatchCurveResult::Failed));
                            return;
                        }
                        std::thread::sleep(Duration::from_millis(100));
                    }
                });
            }
            _ => {}
        });
    }
}
