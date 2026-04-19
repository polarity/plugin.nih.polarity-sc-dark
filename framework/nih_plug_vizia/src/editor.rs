//! The [`Editor`] trait implementation for Vizia editors.

use baseview::{WindowHandle, WindowScalePolicy};
use crossbeam::atomic::AtomicCell;
use nih_plug::debug::*;
use nih_plug::prelude::{Editor, GuiContext, ParentWindowHandle};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use vizia::context::backend::TextConfig;
use vizia::prelude::*;

use crate::widgets::RawParamEvent;
use crate::{assets, widgets, ViziaState, ViziaTheming};

/// An [`Editor`] implementation that calls an vizia draw loop.
pub(crate) struct ViziaEditor {
    pub(crate) vizia_state: Arc<ViziaState>,
    /// The user's app function.
    pub(crate) app: Arc<dyn Fn(&mut Context, Arc<dyn GuiContext>) + 'static + Send + Sync>,
    /// What level of theming to apply. See [`ViziaEditorTheming`].
    pub(crate) theming: ViziaTheming,

    /// The scaling factor reported by the host, if any. On macOS this will never be set and we
    /// should use the system scaling factor instead.
    pub(crate) scaling_factor: AtomicCell<Option<f32>>,

    /// Whether to emit a parameters changed event during the next idle callback. This is set in the
    /// `parameter_values_changed()` implementation and it can be used by widgets to explicitly
    /// check for new parameter values. This is useful when the parameter value is (indirectly) used
    /// to compute a property in an event handler. Like when positioning an element based on the
    /// display value's width.
    pub(crate) emit_parameters_changed_event: Arc<AtomicBool>,
}

impl Editor for ViziaEditor {
    fn spawn(
        &self,
        parent: ParentWindowHandle,
        context: Arc<dyn GuiContext>,
    ) -> Box<dyn std::any::Any + Send> {
        let app = self.app.clone();
        let vizia_state = self.vizia_state.clone();
        let theming = self.theming;

        let (unscaled_width, unscaled_height) = vizia_state.inner_logical_size();
        let system_scaling_factor = self.scaling_factor.load();
        let user_scale_factor = vizia_state.user_scale_factor();

        let mut application = Application::new(move |cx| {
            // Set some default styles to match the iced integration
            if theming >= ViziaTheming::Custom {
                cx.set_default_font(&[assets::NOTO_SANS]);
                if let Err(err) = cx.add_stylesheet(include_style!("assets/theme.css")) {
                    nih_error!("Failed to load stylesheet: {err:?}")
                }

                // There doesn't seem to be any way to bundle styles with a widget, so we'll always
                // include the style sheet for our custom widgets at context creation
                widgets::register_theme(cx);
            }

            // Any widget can change the parameters by emitting `ParamEvent` events. This model will
            // handle them automatically.
            widgets::ParamModel {
                context: context.clone(),
            }
            .build(cx);

            // And we'll link `WindowEvent::ResizeWindow` and `WindowEvent::SetScale` events to our
            // `ViziaState`. We'll notify the host when any of these change.
            let current_inner_window_size = cx.window_size();
            widgets::WindowModel {
                context: context.clone(),
                vizia_state: vizia_state.clone(),
                last_inner_window_size: AtomicCell::new((
                    current_inner_window_size.width,
                    current_inner_window_size.height,
                )),
            }
            .build(cx);

            // Work around a bug in vizia's hover_system where the OVER pseudo-class on the
            // root entity can get permanently cleared when the mouse cursor moves outside the
            // window bounds. This happens when baseview's SetCapture is active (after any
            // mouse-down) and the user drags outside the window: vizia's hover_entity sees the
            // out-of-bounds coordinates and clears OVER on root. Once cleared, hover_system
            // returns early on every subsequent call, permanently breaking hover tracking and
            // making the GUI unresponsive. On Win32, baseview never sends CursorEntered /
            // CursorLeft events, so the OVER flag is never restored.
            //
            // This global listener detects the stuck state on any MouseMove event and emits a
            // synthetic MouseEnter to restore OVER. The next MouseMove after that will then be
            // processed normally by hover_system.
            cx.add_global_listener(|cx, event| {
                event.map(|window_event, _meta| {
                    if let WindowEvent::MouseMove(x, y) = window_event {
                        // The global listener's current entity is Entity::root(), so
                        // is_over() checks root's OVER flag and bounds() returns root bounds.
                        if !cx.is_over() {
                            let bounds = cx.bounds();
                            if *x >= bounds.x
                                && *x < bounds.x + bounds.w
                                && *y >= bounds.y
                                && *y < bounds.y + bounds.h
                            {
                                cx.emit_custom(
                                    Event::new(WindowEvent::MouseEnter)
                                        .origin(Entity::root())
                                        .target(Entity::root())
                                        .propagate(Propagation::Direct),
                                );
                            }
                        }
                    }
                });
            });

            app(cx, context.clone());
        })
        .with_scale_policy(
            system_scaling_factor
                .map(|factor| WindowScalePolicy::ScaleFactor(factor as f64))
                .unwrap_or(WindowScalePolicy::SystemScaleFactor),
        )
        .inner_size((unscaled_width, unscaled_height))
        .user_scale_factor(user_scale_factor)
        .with_text_config(TextConfig {
            hint: false,
            subpixel: true,
        })
        .on_idle({
            let emit_parameters_changed_event = self.emit_parameters_changed_event.clone();
            move |cx| {
                if emit_parameters_changed_event
                    .compare_exchange(true, false, Ordering::AcqRel, Ordering::Relaxed)
                    .is_ok()
                {
                    cx.emit_custom(
                        Event::new(RawParamEvent::ParametersChanged)
                            .propagate(Propagation::Subtree),
                    );
                }
            }
        });

        // This way the plugin can decide to use none of the built in theming
        if theming == ViziaTheming::None {
            application = application.ignore_default_theme();
        }

        let window = application.open_parented(&parent);

        self.vizia_state.open.store(true, Ordering::Release);
        Box::new(ViziaEditorHandle {
            vizia_state: self.vizia_state.clone(),
            window,
        })
    }

    fn size(&self) -> (u32, u32) {
        // This includes the user scale factor if set, but not any HiDPI scaling
        self.vizia_state.scaled_logical_size()
    }

    fn set_scale_factor(&self, factor: f32) -> bool {
        // If the editor is currently open then the host must not change the current HiDPI scale as
        // we don't have a way to handle that. Ableton Live does this.
        if self.vizia_state.is_open() {
            return false;
        }

        // We're making things a bit more complicated by having both a system scale factor, which is
        // used for HiDPI and also known to the host, and a user scale factor that the user can use
        // to arbitrarily resize the GUI
        self.scaling_factor.store(Some(factor));
        true
    }

    fn param_value_changed(&self, _id: &str, _normalized_value: f32) {
        // This will cause a future idle callback to send a parameters changed event.
        // NOTE: We could add an event containing the parameter's ID and the normalized value, but
        //       these events aren't really necessary for Vizia.
        self.emit_parameters_changed_event
            .store(true, Ordering::Relaxed);
    }

    fn param_modulation_changed(&self, _id: &str, _modulation_offset: f32) {
        self.emit_parameters_changed_event
            .store(true, Ordering::Relaxed);
    }

    fn param_values_changed(&self) {
        self.emit_parameters_changed_event
            .store(true, Ordering::Relaxed);
    }
}

/// The window handle used for [`ViziaEditor`].
struct ViziaEditorHandle {
    vizia_state: Arc<ViziaState>,
    window: WindowHandle,
}

/// The window handle enum stored within 'WindowHandle' contains raw pointers. Is there a way around
/// having this requirement?
unsafe impl Send for ViziaEditorHandle {}

impl Drop for ViziaEditorHandle {
    fn drop(&mut self) {
        self.vizia_state.open.store(false, Ordering::Release);
        // On some hosts the parented window may already have been destroyed before the editor
        // handle gets dropped. `baseview::WindowHandle::close()` would then post to a stale HWND.
        if self.window.is_open() {
            self.window.close();
        }
    }
}
