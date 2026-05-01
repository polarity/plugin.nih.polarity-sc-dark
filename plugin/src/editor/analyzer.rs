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
use nih_plug::nih_debug_assert;
use nih_plug::prelude::Param;
use nih_plug_vizia::assets;
use nih_plug_vizia::vizia::prelude::*;
use nih_plug_vizia::vizia::vg;
use nih_plug_vizia::widgets::RawParamEvent;
use std::cell::Cell;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::analyzer::AnalyzerData;
use crate::curve::{
    Curve, CurvePoint, THRESHOLD_CURVE_MAX_FREQUENCY_HZ, THRESHOLD_CURVE_MIN_FREQUENCY_HZ,
    THRESHOLD_CURVE_POINT_OFFSET_LIMIT_DB,
};
use crate::SpectralCompressorParams;

use super::theme;

const LN_FREQ_RANGE_START_HZ: f32 = 3.4011974; // 30.0f32.ln();
const LN_FREQ_RANGE_END_HZ: f32 = 9.998797; // 22_000.0f32.ln();
const LN_FREQ_RANGE: f32 = LN_FREQ_RANGE_END_HZ - LN_FREQ_RANGE_START_HZ;
const FREQUENCY_GUIDES: [(f32, &str); 4] = [
    (30.0, "30 Hz"),
    (100.0, "100 Hz"),
    (1_000.0, "1 kHz"),
    (10_000.0, "10 kHz"),
];

// All analyzer colors are defined in `theme.rs`.

/// A very analyzer showing the envelope followers as a magnitude spectrum with an overlay for the
/// gain reduction.
pub struct Analyzer {
    analyzer_data: Arc<AtomicRefCell<triple_buffer::Output<AnalyzerData>>>,
    sample_rate: Arc<AtomicF32>,
    params: Arc<SpectralCompressorParams>,
    frequency_guide_label_font: Cell<Option<vg::FontId>>,
    selected_point_index: Option<usize>,
    drag_active: bool,
}

impl Analyzer {
    /// Creates a new [`Analyzer`].
    pub fn new<LAnalyzerData, LRate>(
        cx: &mut Context,
        analyzer_data: LAnalyzerData,
        sample_rate: LRate,
        params: impl Lens<Target = Arc<SpectralCompressorParams>>,
    ) -> Handle<'_, Self>
    where
        LAnalyzerData: Lens<Target = Arc<AtomicRefCell<triple_buffer::Output<AnalyzerData>>>>,
        LRate: Lens<Target = Arc<AtomicF32>>,
    {
        Self {
            analyzer_data: analyzer_data.get(cx),
            sample_rate: sample_rate.get(cx),
            params: params.get(cx),
            frequency_guide_label_font: Cell::new(None),
            selected_point_index: None,
            drag_active: false,
        }
        .build(
            cx,
            // This is an otherwise empty element only used for custom drawing
            |_cx| (),
        )
    }

    fn begin_drag(&mut self, cx: &mut EventContext, point_index: usize) {
        self.selected_point_index = Some(point_index);
        self.drag_active = true;

        let point = &self.params.threshold.curve_points[point_index];
        Self::emit_begin(cx, &point.frequency);
        Self::emit_begin(cx, &point.offset_db);
    }

    fn finish_drag(&mut self, cx: &mut EventContext) {
        if let Some(point_index) = self.selected_point_index {
            let point = &self.params.threshold.curve_points[point_index];
            Self::emit_end(cx, &point.frequency);
            Self::emit_end(cx, &point.offset_db);
        }

        self.selected_point_index = None;
        self.drag_active = false;
        cx.release();
        cx.set_active(false);
    }

    fn first_free_point_index(&self) -> Option<usize> {
        self.params
            .threshold
            .curve_points
            .iter()
            .position(|point| !point.enabled.value())
    }

    fn set_point_enabled(&self, cx: &mut EventContext, point_index: usize, enabled: bool) {
        let point = &self.params.threshold.curve_points[point_index];
        Self::emit_begin(cx, &point.enabled);
        Self::emit_set(cx, &point.enabled, enabled);
        Self::emit_end(cx, &point.enabled);
    }

    fn set_point_position(
        &self,
        cx: &mut EventContext,
        point_index: usize,
        frequency: f32,
        offset_db: f32,
    ) {
        let point = &self.params.threshold.curve_points[point_index];
        Self::emit_set(cx, &point.frequency, frequency);
        Self::emit_set(cx, &point.offset_db, offset_db);
    }

    fn hit_test_point(
        &self,
        bounds: BoundingBox,
        x: f32,
        y: f32,
        scale_factor: f32,
    ) -> Option<usize> {
        if bounds.w <= 0.0 || bounds.h <= 0.0 {
            return None;
        }

        let curve_params = self.params.threshold.curve_params();
        let curve = Curve::new(&curve_params);
        let display_offset_db = self.editable_curve_display_offset_db();
        let hit_radius = scale_factor * 8.0;
        let max_distance_squared = hit_radius * hit_radius;
        let mut nearest = None;

        for (index, point) in curve_params.points.iter().enumerate() {
            let Some((point_x, point_y)) =
                point_screen_position(bounds, &curve, point, display_offset_db)
            else {
                continue;
            };

            let dx = x - point_x;
            let dy = y - point_y;
            let distance_squared = (dx * dx) + (dy * dy);
            if distance_squared <= max_distance_squared
                && nearest
                    .map(|(_, nearest_distance_squared)| {
                        distance_squared < nearest_distance_squared
                    })
                    .unwrap_or(true)
            {
                nearest = Some((index, distance_squared));
            }
        }

        nearest.map(|(index, _)| index)
    }

    fn screen_to_point_values(&self, bounds: BoundingBox, x: f32, y: f32) -> (f32, f32) {
        let x_t = if bounds.w <= 0.0 {
            0.0
        } else {
            ((x - bounds.x) / bounds.w).clamp(0.0, 1.0)
        };
        let y_t = if bounds.h <= 0.0 {
            0.0
        } else {
            (1.0 - ((y - bounds.y) / bounds.h)).clamp(0.0, 1.0)
        };

        let ln_freq = LN_FREQ_RANGE_START_HZ + (LN_FREQ_RANGE * x_t);
        let frequency = ln_freq.exp().clamp(
            THRESHOLD_CURVE_MIN_FREQUENCY_HZ,
            THRESHOLD_CURVE_MAX_FREQUENCY_HZ,
        );
        let clicked_db = unclamped_t_to_db(y_t);
        let curve_params = self.params.threshold.curve_params();
        let curve = Curve::new(&curve_params);
        let offset_db = (clicked_db
            - self.editable_curve_display_offset_db()
            - curve.evaluate_base_ln(ln_freq))
        .clamp(
            -THRESHOLD_CURVE_POINT_OFFSET_LIMIT_DB,
            THRESHOLD_CURVE_POINT_OFFSET_LIMIT_DB,
        );

        (frequency, offset_db)
    }

    fn editable_curve_display_offset_db(&self) -> f32 {
        let upwards_offset_db = self.params.compressors.upwards.threshold_offset_db.value();
        let downwards_offset_db = self
            .params
            .compressors
            .downwards
            .threshold_offset_db
            .value();

        (upwards_offset_db + downwards_offset_db) * 0.5
    }

    fn emit_begin<P: Param>(cx: &mut EventContext, param: &P) {
        cx.emit(RawParamEvent::BeginSetParameter(param.as_ptr()));
    }

    fn emit_set<P: Param>(cx: &mut EventContext, param: &P, value: P::Plain) {
        cx.emit(RawParamEvent::SetParameterNormalized(
            param.as_ptr(),
            param.preview_normalized(value),
        ));
    }

    fn emit_end<P: Param>(cx: &mut EventContext, param: &P) {
        cx.emit(RawParamEvent::EndSetParameter(param.as_ptr()));
    }
}

impl View for Analyzer {
    fn element(&self) -> Option<&'static str> {
        Some("analyzer")
    }

    fn event(&mut self, cx: &mut EventContext, event: &mut Event) {
        event.map(|window_event, meta| match window_event {
            WindowEvent::MouseDown(MouseButton::Left)
            | WindowEvent::MouseDoubleClick(MouseButton::Left)
            | WindowEvent::MouseTripleClick(MouseButton::Left) => {
                if self.drag_active {
                    self.finish_drag(cx);
                    meta.consume();
                    return;
                }

                let bounds = cx.bounds();
                let x = cx.mouse().cursorx;
                let y = cx.mouse().cursory;

                if let Some(point_index) = self.hit_test_point(bounds, x, y, cx.scale_factor()) {
                    self.begin_drag(cx, point_index);
                } else if let Some(point_index) = self.first_free_point_index() {
                    let (frequency, offset_db) = self.screen_to_point_values(bounds, x, y);
                    self.set_point_enabled(cx, point_index, true);
                    self.begin_drag(cx, point_index);
                    self.set_point_position(cx, point_index, frequency, offset_db);
                } else {
                    meta.consume();
                    return;
                }

                cx.capture();
                cx.focus();
                cx.set_active(true);
                meta.consume();
            }
            WindowEvent::MouseDown(MouseButton::Right)
            | WindowEvent::MouseDoubleClick(MouseButton::Right)
            | WindowEvent::MouseTripleClick(MouseButton::Right) => {
                if self.drag_active {
                    self.finish_drag(cx);
                    meta.consume();
                    return;
                }

                let bounds = cx.bounds();
                let x = cx.mouse().cursorx;
                let y = cx.mouse().cursory;

                if let Some(point_index) = self.hit_test_point(bounds, x, y, cx.scale_factor()) {
                    self.set_point_enabled(cx, point_index, false);
                    meta.consume();
                }
            }
            WindowEvent::MouseUp(MouseButton::Left) => {
                if self.drag_active {
                    self.finish_drag(cx);
                    meta.consume();
                }
            }
            WindowEvent::MouseMove(x, y) => {
                if self.drag_active {
                    if let Some(point_index) = self.selected_point_index {
                        let bounds = cx.bounds();
                        let (frequency, offset_db) = self.screen_to_point_values(bounds, *x, *y);
                        self.set_point_position(cx, point_index, frequency, offset_db);
                    }

                    meta.consume();
                }
            }
            _ => {}
        });
    }

    fn draw(&self, cx: &mut DrawContext, canvas: &mut Canvas) {
        let bounds = cx.bounds();
        if bounds.w == 0.0 || bounds.h == 0.0 {
            return;
        }

        // Fill the analyzer background explicitly since custom canvas drawing
        // bypasses the CSS background-color property
        let mut bg_path = vg::Path::new();
        bg_path.rect(bounds.x, bounds.y, bounds.w, bounds.h);
        canvas.fill_path(&bg_path, &vg::Paint::color(theme::ANALYZER_BACKGROUND));

        draw_frequency_guide_lines(cx, canvas);

        // The analyzer data is pulled directly from the spectral `CompressorBank`
        let Ok(mut analyzer_data) = self.analyzer_data.try_borrow_mut() else {
            return;
        };
        let analyzer_data = analyzer_data.read();
        let nyquist = self.sample_rate.load(Ordering::Relaxed) / 2.0;

        draw_spectrum(cx, canvas, analyzer_data, nyquist);
        draw_threshold_curve(cx, canvas, analyzer_data);
        draw_gain_reduction(cx, canvas, analyzer_data, nyquist);
        let label_font = self.frequency_guide_label_font.get().or_else(|| {
            canvas
                .add_font_mem(assets::fonts::NOTO_SANS_LIGHT)
                .ok()
                .inspect(|font_id| self.frequency_guide_label_font.set(Some(*font_id)))
        });
        draw_frequency_guide_labels(cx, canvas, label_font);

        // Draw the border last
        let border_width = cx.border_width();
        let border_color: vg::Color = cx.border_color().into();

        let mut path = vg::Path::new();
        {
            let x = bounds.x + border_width / 2.0;
            let y = bounds.y + border_width / 2.0;
            let w = bounds.w - border_width;
            let h = bounds.h - border_width;
            path.move_to(x, y);
            path.line_to(x, y + h);
            path.line_to(x + w, y + h);
            path.line_to(x + w, y);
            path.close();
        }

        let paint = vg::Paint::color(border_color).with_line_width(border_width);
        canvas.stroke_path(&path, &paint);
    }
}

/// Compute an unclamped value based on a decibel value -80 and is mapped to 0, +20 is mapped to 1,
/// and all other values are linearly interpolated from there
#[inline]
fn db_to_unclamped_t(db_value: f32) -> f32 {
    (db_value + 80.0) / 100.0
}

#[inline]
fn unclamped_t_to_db(t: f32) -> f32 {
    (t * 100.0) - 80.0
}

fn editable_curve_display_offset_db(analyzer_data: &AnalyzerData) -> f32 {
    let (upwards_offset_db, downwards_offset_db) = analyzer_data.curve_offsets_db;
    (upwards_offset_db + downwards_offset_db) * 0.5
}

#[inline]
fn frequency_to_x_t(frequency_hz: f32) -> f32 {
    ln_frequency_to_x_t(frequency_hz.ln())
}

#[inline]
fn ln_frequency_to_x_t(ln_frequency: f32) -> f32 {
    (ln_frequency - LN_FREQ_RANGE_START_HZ) / LN_FREQ_RANGE
}

fn draw_frequency_guide_lines(cx: &mut DrawContext, canvas: &mut Canvas) {
    let bounds = cx.bounds();
    let scale_factor = cx.scale_factor();
    let line_width = scale_factor;
    let line_paint =
        vg::Paint::color(theme::ANALYZER_FREQUENCY_GUIDE_LINE).with_line_width(line_width);

    canvas.scissor(bounds.x, bounds.y, bounds.w, bounds.h);
    for (frequency_hz, _) in FREQUENCY_GUIDES {
        let x_t = frequency_to_x_t(frequency_hz);
        if !(0.0..=1.0).contains(&x_t) {
            continue;
        }

        let x = bounds.x + (bounds.w * x_t);

        let mut path = vg::Path::new();
        path.move_to(x, bounds.y);
        path.line_to(x, bounds.y + bounds.h);
        canvas.stroke_path(&path, &line_paint);
    }
    canvas.reset_scissor();
}

fn draw_frequency_guide_labels(
    cx: &mut DrawContext,
    canvas: &mut Canvas,
    font_id: Option<vg::FontId>,
) {
    let bounds = cx.bounds();
    let scale_factor = cx.scale_factor();
    let label_padding = scale_factor * 5.0;
    let label_y = bounds.y + label_padding;
    let label_paint = vg::Paint::color(theme::ANALYZER_FREQUENCY_GUIDE_LABEL)
        .with_font_size(scale_factor * 11.0)
        .with_text_baseline(vg::Baseline::Top);
    let label_paint = if let Some(font_id) = font_id {
        label_paint.with_font(&[font_id])
    } else {
        label_paint
    };

    canvas.scissor(bounds.x, bounds.y, bounds.w, bounds.h);
    for (frequency_hz, label) in FREQUENCY_GUIDES {
        let x_t = frequency_to_x_t(frequency_hz);
        if !(0.0..=1.0).contains(&x_t) {
            continue;
        }

        let line_x = bounds.x + (bounds.w * x_t);
        let min_label_x = bounds.x + label_padding;
        let max_label_x = bounds.x + bounds.w - label_padding;
        let label_x = if min_label_x <= max_label_x {
            (line_x + label_padding).clamp(min_label_x, max_label_x)
        } else {
            bounds.x + (bounds.w * 0.5)
        };
        let label_paint = label_paint.clone().with_text_align(vg::Align::Left);

        let _ = canvas.fill_text(label_x, label_y, label, &label_paint);
    }
    canvas.reset_scissor();
}

fn point_screen_position(
    bounds: BoundingBox,
    curve: &Curve,
    point: &CurvePoint,
    display_offset_db: f32,
) -> Option<(f32, f32)> {
    if !point.enabled || !point.frequency.is_finite() {
        return None;
    }

    let ln_freq = point
        .frequency
        .clamp(
            THRESHOLD_CURVE_MIN_FREQUENCY_HZ,
            THRESHOLD_CURVE_MAX_FREQUENCY_HZ,
        )
        .ln();
    let x_t = ln_frequency_to_x_t(ln_freq);
    if !(0.0..=1.0).contains(&x_t) {
        return None;
    }

    let y_db = curve.evaluate_ln(ln_freq) + display_offset_db;
    let y_t = db_to_unclamped_t(y_db);

    Some((
        bounds.x + (bounds.w * x_t),
        bounds.y + (bounds.h * (1.0 - y_t)),
    ))
}

/// Draw the spectrum analyzer part of the analyzer. These are drawn as vertical bars until the
/// spacing between the bars becomes less the line width, at which point it's drawn as a solid mesh
/// instead.
fn draw_spectrum(
    cx: &mut DrawContext,
    canvas: &mut Canvas,
    analyzer_data: &AnalyzerData,
    nyquist_hz: f32,
) {
    let bounds = cx.bounds();

    let line_width = cx.scale_factor() * 1.5;
    let spectrum_color = theme::ANALYZER_SPECTRUM_BARS;
    // This is used to draw the individual bars
    let bars_paint = vg::Paint::color(spectrum_color).with_line_width(line_width);
    // These colors are used to draw the mesh part of the spectrum. A gradient fades from the
    // lighter variant on the left to the darker variant on the right.
    let lighter_spectrum_color = theme::ANALYZER_SPECTRUM_MESH_LIGHT;
    let darker_spectrum_color = theme::ANALYZER_SPECTRUM_MESH_DARK;

    // The frequency belonging to a bin in Hz
    let bin_frequency = |bin_idx: f32| (bin_idx / analyzer_data.num_bins as f32) * nyquist_hz;
    // A `[0, 1]` value indicating at which relative x-coordinate a bin should be drawn at
    let bin_t =
        |bin_idx: f32| (bin_frequency(bin_idx).ln() - LN_FREQ_RANGE_START_HZ) / LN_FREQ_RANGE;
    // Converts a linear magnitude value in to a `[0, 1]` value where 0 is -80 dB or lower, and 1 is
    // +20 dB or higher.
    let magnitude_height = |magnitude: f32| {
        nih_debug_assert!(magnitude >= 0.0);
        let magnitude_db = nih_plug::util::gain_to_db(magnitude);
        db_to_unclamped_t(magnitude_db).clamp(0.0, 1.0)
    };

    // The first part of this drawing routing is simple. Individual bins are drawn as bars until the
    // distance between the bars approaches `mesh_start_delta_threshold`. After that the rest is
    // drawn as a solid mesh.
    let mesh_start_delta_threshold = line_width + 0.5;
    let mut mesh_bin_start_idx = analyzer_data.num_bins;
    let mut previous_physical_x_coord = bounds.x - 2.0;

    let mut bars_path = vg::Path::new();
    for (bin_idx, magnitude) in analyzer_data
        .envelope_followers
        .iter()
        .enumerate()
        .take(analyzer_data.num_bins)
    {
        let t = bin_t(bin_idx as f32);
        if t <= 0.0 || t >= 1.0 {
            continue;
        }

        let physical_x_coord = bounds.x + (bounds.w * t);
        if physical_x_coord - previous_physical_x_coord < mesh_start_delta_threshold {
            // NOTE: We'll draw this one bar earlier because we're not stroking the solid mesh part,
            //       and otherwise there would be a weird looking gap at the left side
            mesh_bin_start_idx = bin_idx.saturating_sub(1);
            previous_physical_x_coord = physical_x_coord;
            break;
        }

        // Scale this so that 1.0/0 dBFS magnitude is at 80% of the height, the bars begin
        // at -80 dBFS, and that the scaling is linear. This is the same scaling used in
        // Diopser's spectrum analyzer.
        let height = magnitude_height(*magnitude);

        bars_path.move_to(physical_x_coord, bounds.y + (bounds.h * (1.0 - height)));
        bars_path.line_to(physical_x_coord, bounds.y + bounds.h);

        previous_physical_x_coord = physical_x_coord;
    }
    canvas.stroke_path(&bars_path, &bars_paint);

    // The mesh path starts at the bottom left, follows the top envelope of the spectrum analyzer,
    // and ends in the bottom right
    let mut mesh_path = vg::Path::new();
    let mesh_start_x_coordiante = bounds.x + (bounds.w * bin_t(mesh_bin_start_idx as f32));
    let mesh_start_y_coordinate = bounds.y + bounds.h;

    mesh_path.move_to(mesh_start_x_coordiante, mesh_start_y_coordinate);
    for (bin_idx, magnitude) in analyzer_data
        .envelope_followers
        .iter()
        .enumerate()
        .take(analyzer_data.num_bins)
        .skip(mesh_bin_start_idx)
    {
        let t = bin_t(bin_idx as f32);
        if t <= 0.0 || t >= 1.0 {
            continue;
        }

        let physical_x_coord = bounds.x + (bounds.w * t);
        previous_physical_x_coord = physical_x_coord;
        let height = magnitude_height(*magnitude);
        if height > 0.0 {
            mesh_path.line_to(
                physical_x_coord,
                // This includes the line width, since this path is not stroked
                bounds.y + (bounds.h * (1.0 - height) - (line_width / 2.0)).max(0.0),
            );
        } else {
            mesh_path.line_to(physical_x_coord, mesh_start_y_coordinate);
        }
    }

    mesh_path.line_to(previous_physical_x_coord, mesh_start_y_coordinate);
    mesh_path.close();

    let mesh_paint = vg::Paint::linear_gradient_stops(
        mesh_start_x_coordiante,
        0.0,
        previous_physical_x_coord,
        0.0,
        [
            (0.0, lighter_spectrum_color),
            (0.707, darker_spectrum_color),
            (1.0, darker_spectrum_color),
        ],
    )
    .with_anti_alias(false);
    canvas.fill_path(&mesh_path, &mesh_paint);
}

/// Overlays the threshold curve over the spectrum analyzer. If either the upwards or downwards
/// threshold offsets are non-zero then two curves are drawn.
fn draw_threshold_curve(cx: &mut DrawContext, canvas: &mut Canvas, analyzer_data: &AnalyzerData) {
    let bounds = cx.bounds();

    let line_width = cx.scale_factor() * 3.0;
    let downwards_paint =
        vg::Paint::color(theme::ANALYZER_THRESHOLD_DOWNWARDS).with_line_width(line_width);
    let upwards_paint =
        vg::Paint::color(theme::ANALYZER_THRESHOLD_UPWARDS).with_line_width(line_width);

    // This can be done slightly cleverer but for our purposes drawing line segments that are either
    // 1 pixel apart or that split the curve up into 100 segments (whichever results in the least
    // amount of line segments) should be sufficient
    let curve = Curve::new(&analyzer_data.curve_params);
    let num_points = 100.min(bounds.w.ceil() as usize);

    let mut draw_with_offset = |offset_db: f32, paint: vg::Paint| {
        let mut path = vg::Path::new();
        for i in 0..num_points {
            let x_t = i as f32 / (num_points - 1) as f32;
            let ln_freq = LN_FREQ_RANGE_START_HZ + (LN_FREQ_RANGE * x_t);

            // Evaluating the curve results in a value in dB, which must then be mapped to the same
            // scale used in `draw_spectrum()`
            let y_db = curve.evaluate_ln(ln_freq) + offset_db;
            let y_t = db_to_unclamped_t(y_db);

            let physical_x_pos = bounds.x + (bounds.w * x_t);
            // This value increases from bottom to top
            let physical_y_pos = bounds.y + (bounds.h * (1.0 - y_t));

            if i == 0 {
                path.move_to(physical_x_pos, physical_y_pos);
            } else {
                path.line_to(physical_x_pos, physical_y_pos);
            }
        }

        // This does a way better job at cutting off the tops and bottoms of the graph than we could do
        // by hand
        canvas.scissor(bounds.x, bounds.y, bounds.w, bounds.h);
        canvas.stroke_path(&path, &paint);
        canvas.reset_scissor();
    };

    let (upwards_offset_db, downwards_offset_db) = analyzer_data.curve_offsets_db;
    draw_with_offset(upwards_offset_db, upwards_paint);
    draw_with_offset(downwards_offset_db, downwards_paint);
    draw_curve_points(cx, canvas, analyzer_data, &curve);
}

fn draw_curve_points(
    cx: &mut DrawContext,
    canvas: &mut Canvas,
    analyzer_data: &AnalyzerData,
    curve: &Curve,
) {
    let bounds = cx.bounds();
    let radius = cx.scale_factor() * 4.5;
    let stroke_width = cx.scale_factor() * 1.5;
    let display_offset_db = editable_curve_display_offset_db(analyzer_data);
    let fill_paint = vg::Paint::color(theme::ANALYZER_THRESHOLD_POINT_FILL);
    let stroke_paint =
        vg::Paint::color(theme::ANALYZER_THRESHOLD_POINT_STROKE).with_line_width(stroke_width);

    canvas.scissor(bounds.x, bounds.y, bounds.w, bounds.h);
    for point in &analyzer_data.curve_params.points {
        let Some((x, y)) = point_screen_position(bounds, curve, point, display_offset_db) else {
            continue;
        };

        let mut path = vg::Path::new();
        path.circle(x, y, radius);
        canvas.fill_path(&path, &fill_paint);
        canvas.stroke_path(&path, &stroke_paint);
    }
    canvas.reset_scissor();
}

/// Overlays the gain reduction display over the spectrum analyzer.
/// Downwards compression (negative gain_difference_db) is drawn in yellow,
/// upwards expansion (positive gain_difference_db) is drawn in blue.
fn draw_gain_reduction(
    cx: &mut DrawContext,
    canvas: &mut Canvas,
    analyzer_data: &AnalyzerData,
    nyquist_hz: f32,
) {
    let bounds = cx.bounds();

    let downwards_paint = vg::Paint::color(theme::ANALYZER_GR_DOWNWARDS).with_anti_alias(false);
    let upwards_paint = vg::Paint::color(theme::ANALYZER_GR_UPWARDS).with_anti_alias(false);

    let bin_frequency = |bin_idx: f32| (bin_idx / analyzer_data.num_bins as f32) * nyquist_hz;

    let mut downwards_path = vg::Path::new();
    let mut upwards_path = vg::Path::new();

    for (bin_idx, &gain_db) in analyzer_data
        .gain_difference_db
        .iter()
        .enumerate()
        .take(analyzer_data.num_bins)
    {
        // Avoid drawing tiny slivers for low gain values
        if gain_db.abs() < 0.2 {
            continue;
        }

        // The gain reduction bars are drawn with the width of the bin, centered on the bin's center
        // frequency. The first and the last bin are extended to the edges of the graph because
        // otherwise it looks weird.
        let t_start = if bin_idx == 0 {
            0.0
        } else {
            let gr_start_ln_frequency = bin_frequency(bin_idx as f32 - 0.5).ln();
            (gr_start_ln_frequency - LN_FREQ_RANGE_START_HZ) / LN_FREQ_RANGE
        };
        let t_end = if bin_idx == analyzer_data.num_bins - 1 {
            1.0
        } else {
            let gr_end_ln_frequency = bin_frequency(bin_idx as f32 + 0.5).ln();
            (gr_end_ln_frequency - LN_FREQ_RANGE_START_HZ) / LN_FREQ_RANGE
        };
        if t_end < 0.0 || t_start > 1.0 {
            continue;
        }

        let (t_start, t_end) = (t_start.max(0.0), t_end.min(1.0));

        // For the bar's height we'll draw 0 dB of gain reduction as a flat line (except we
        // don't actually draw 0 dBs of GR because it looks glitchy, but that's besides the
        // point). 40 dB of gain reduction causes the bar to be drawn from the center all
        // the way to the bottom of the spectrum analyzer. 40 dB of additional gain causes
        // the bar to be drawn from the center all the way to the top of the graph.
        // NOTE: Y-coordinates go from top to bottom, hence the minus
        let t_y = ((-gain_db + 40.0) / 80.0).clamp(0.0, 1.0);

        let path = if gain_db < 0.0 {
            &mut downwards_path
        } else {
            &mut upwards_path
        };

        path.move_to(bounds.x + (bounds.w * t_start), bounds.y + (bounds.h * 0.5));
        path.line_to(bounds.x + (bounds.w * t_end), bounds.y + (bounds.h * 0.5));
        path.line_to(bounds.x + (bounds.w * t_end), bounds.y + (bounds.h * t_y));
        path.line_to(bounds.x + (bounds.w * t_start), bounds.y + (bounds.h * t_y));
        path.close();
    }

    // Use standard alpha blending so the overlay looks correct on dark backgrounds
    canvas
        .global_composite_blend_func(vg::BlendFactor::SrcAlpha, vg::BlendFactor::OneMinusSrcAlpha);
    canvas.fill_path(&downwards_path, &downwards_paint);
    canvas.fill_path(&upwards_path, &upwards_paint);
    canvas.global_composite_blend_func(vg::BlendFactor::One, vg::BlendFactor::OneMinusSrcAlpha);
}
