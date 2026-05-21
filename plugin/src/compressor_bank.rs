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

use nih_plug::prelude::*;
use realfft::num_complex::Complex32;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::analyzer::AnalyzerData;
use crate::curve::{
    Curve, CurveParams, CurvePoint, MAX_THRESHOLD_CURVE_POINTS, THRESHOLD_CURVE_MAX_FREQUENCY_HZ,
    THRESHOLD_CURVE_MIN_FREQUENCY_HZ, THRESHOLD_CURVE_POINT_OFFSET_LIMIT_DB,
};
use crate::frozen_ir::FrozenIrData;
use crate::match_curve::{MatchCurveMeter, MatchCurveRuntime};
use crate::SpectralCompressorParams;

// These are the parameter name prefixes used for the downwards and upwards compression parameters.
// The ID prefixes a re set in the `CompressorBankParams` struct.
const DOWNWARDS_NAME_PREFIX: &str = "Downwards";
const UPWARDS_NAME_PREFIX: &str = "Upwards";
const GAIN_SMOOTHING_MAX_RADIUS_LN: f32 = std::f32::consts::LN_2;
const PINK_NOISE_SLOPE_OFFSET_DB_PER_OCT: f32 = 3.0;

/// The envelopes are initialized to the RMS value of a -24 dB sine wave to make sure extreme upwards
/// compression doesn't cause pops when switching between window sizes and when deactivating and
/// reactivating the plugin.
const ENVELOPE_INIT_VALUE: f32 = std::f32::consts::FRAC_1_SQRT_2 / 8.0;

/// The target frequency for the high frequency ratio rolloff. This is fixed to prevent Spectral
/// Compressor from getting brighter as the sample rate increases.
const HIGH_FREQ_RATIO_ROLLOFF_FREQUENCY_LN: f32 = 10.001068; // 22_050.0f32.ln()

/// The length of time over which the envelope followers fade back from being instant to using the
/// configured timingsafter the compressor bank has been reset.
const ENVELOPE_FOLLOWER_TIMING_FADE_MS: f32 = 150.0;

/// A bank of compressors so each FFT bin can be compressed individually. The vectors in this struct
/// will have a capacity of `MAX_WINDOW_SIZE / 2 + 1` and a size that matches the current complex
/// FFT buffer size. This is stored as a struct of arrays to make SIMD-ing easier in the future.
pub struct CompressorBank {
    /// If set, then the downwards thresholds should be updated on the next processing cycle. Can be
    /// set from a parameter value change listener, and is also set when calling `.reset_for_size`.
    pub should_update_downwards_thresholds: Arc<AtomicBool>,
    /// The same as `should_update_downwards_thresholds`, but for upwards thresholds.
    pub should_update_upwards_thresholds: Arc<AtomicBool>,
    /// If set, then the downwards ratios should be updated on the next processing cycle. Can be set
    /// from a parameter value change listener, and is also set when calling `.reset_for_size`.
    pub should_update_downwards_ratios: Arc<AtomicBool>,
    /// The same as `should_update_downwards_ratios`, but for upwards ratios.
    pub should_update_upwards_ratios: Arc<AtomicBool>,
    /// If set, then the parameters for the downwards compression soft knee parabola should be
    /// updated on the next processing cycle. Can be set from a parameter value change listener, and
    /// is also set when calling `.reset_for_size`.
    pub should_update_downwards_knee_parabolas: Arc<AtomicBool>,
    /// The same as `should_update_downwards_knee_parabolas`, but for upwards compression.
    pub should_update_upwards_knee_parabolas: Arc<AtomicBool>,

    /// For each compressor bin, `ln(freq)` where `freq` is the frequency associated with that
    /// compressor. This is precomputed since all update functions need it.
    ln_freqs: Vec<f32>,

    /// Downwards compressor thresholds, in decibels.
    downwards_thresholds_db: Vec<f32>,
    /// The ratios for the the downwards compressors. At 1.0 the compressors won't do anything. If
    /// [`CompressorBankParams::high_freq_ratio_rolloff`] is set to 1.0, then this will be the same
    /// for each compressor.
    downwards_ratios: Vec<f32>,
    /// The knee is modelled as a parabola using the formula `x + a * (x + b)^2`. This is `a` in
    /// that equation. The formula is taken from the Digital Dynamic Range Compressor Design paper
    /// by Dimitrios Giannoulis et. al.
    downwards_knee_parabola_scale: Vec<f32>,
    /// `b` in the equation from `downwards_knee_parabola_scale`.
    downwards_knee_parabola_intercept: Vec<f32>,

    /// Upwards compressor thresholds, in decibels.
    upwards_thresholds_db: Vec<f32>,
    /// The same as `downwards_ratios`, but for the upwards compression.
    upwards_ratios: Vec<f32>,
    /// `downwards_knee_parabola_scale`, but for the upwards compressors.
    upwards_knee_parabola_scale: Vec<f32>,
    /// `downwards_knee_parabola_intercept`, but for the upwards compressors.
    upwards_knee_parabola_intercept: Vec<f32>,

    /// The current envelope value for this bin, in linear space. Indexed by
    /// `[channel_idx][compressor_idx]`.
    envelopes: Vec<Vec<f32>>,
    /// A scaling factor for the envelope follower timings. This is set to 0 and then slowly brought
    /// back up to 1 after after [`CompressorBank::reset()`] has been called to allow the envelope
    /// followers to settle back in.
    envelope_followers_timing_scale: f32,
    /// When sidechaining is enabled, this contains the per-channel frqeuency spectrum magnitudes
    /// for the current block. The compressor thresholds and knee values are multiplied by these
    /// values to get the effective thresholds.
    sidechain_spectrum_magnitudes: Vec<Vec<f32>>,
    /// Per-bin channel-linked sidechain magnitudes for the current block.
    linked_sidechain_magnitudes: Vec<f32>,
    /// Scratch buffer for per-bin gain differences before spectral smoothing.
    raw_gain_difference_db: Vec<f32>,
    /// Scratch buffer for the per-bin gain differences that will be applied.
    smoothed_gain_difference_db: Vec<f32>,
    /// Prefix sums for smoothing the gain curve without per-bin inner summing.
    gain_smoothing_prefix_db: Vec<f32>,
    /// The frozen per-bin gain difference snapshot for each channel. When freeze is enabled these
    /// values are captured once and then reused until freeze is disabled or invalidated.
    frozen_gain_difference_db: Vec<Vec<f32>>,
    /// Whether the frozen gain difference snapshot for a channel has already been captured.
    frozen_gain_snapshot_valid: Vec<bool>,
    /// Tracks whether freeze was active in the previous compressor processing pass.
    freeze_was_active: bool,
    /// Tracks whether the current frozen snapshot has already been published for IR export.
    frozen_ir_snapshot_published: bool,
    /// The window size this compressor bank was configured for. This is used to compute the
    /// coefficients for the envelope followers in the process function.
    window_size: usize,
    /// The sample rate this compressor bank was configured for. This is used to compute the
    /// coefficients for the envelope followers in the process function.
    sample_rate: f32,

    /// The input data for the spectrum analyzer. Stores both the spectrum analyzer values and the
    /// current gain reduction. Used to draw the spectrum analyzer and gain reduction display in the
    /// editor.
    analyzer_input_data: triple_buffer::Input<AnalyzerData>,
    /// The exact frozen compressor gain curve shared with the editor for IR export.
    frozen_ir_input_data: triple_buffer::Input<FrozenIrData>,
    /// Set to `true` when analyzer data has been fully prepared and is ready to publish.
    analyzer_needs_publish: bool,
    /// Request/result handoff for editor-triggered threshold curve matching.
    match_curve_runtime: Arc<MatchCurveRuntime>,
    /// Audio-thread local accumulator for threshold curve matching.
    match_curve_meter: MatchCurveMeter,
}

#[derive(Params)]
pub struct ThresholdParams {
    /// The compressor threshold at the center frequency. When sidechaining is enabled, the input
    /// signal is gained by the inverse of this value. This replaces the input gain in the original
    /// Spectral Compressor. In the polynomial below, this is the intercept.
    #[id = "tresh_global"]
    pub threshold_db: FloatParam,
    /// The center frqeuency for the target curve when sidechaining is not enabled. The curve is a
    /// polynomial `threshold_db + curve_slope*x + curve_curve*(x^2)` that evaluates to a decibel
    /// value, where `x = ln(center_frequency) - ln(bin_frequency)`. In other words, this is
    /// evaluated in the log/log domain for decibels and octaves.
    #[id = "thresh_center_freq"]
    pub center_frequency: FloatParam,
    /// The slope for the curve, in the log/log domain. See the polynomial above.
    #[id = "thresh_curve_slope"]
    pub curve_slope: FloatParam,
    /// The, uh, 'curve' for the curve, in the logarithmic domain. This is the third coefficient in
    /// the quadratic polynomial and controls the parabolic behavior. Positive values turn the curve
    /// into a v-shaped curve, while negative values attenuate everything outside of the center
    /// frequency. See the polynomial above.
    #[id = "thresh_curve_curve"]
    pub curve_curve: FloatParam,
    /// Hidden point slots edited from the analyzer graph. These are regular parameters so plugin
    /// state is saved and restored by hosts without requiring custom serialization.
    #[nested(array)]
    pub curve_points: [ThresholdCurvePointParams; MAX_THRESHOLD_CURVE_POINTS],

    /// Controls the type of threshold that should be used. Check [`ThresholdMode`] for more
    /// information.
    #[id = "thresh_mode"]
    pub mode: EnumParam<ThresholdMode>,
    /// A `[0, 1]` parameter that controls how much of the other channels should be mixed in when
    /// computing the channel gain value that is then multiplied with he thresholds and knee values
    /// to the the compression parameters when using the sidechain modes.
    #[id = "thresh_sc_link"]
    pub sc_channel_link: FloatParam,
    /// Smooths the final per-bin gain reduction curve before applying it to the FFT bins.
    #[id = "gain_smoothing"]
    pub gain_smoothing: FloatParam,
}

#[derive(Params)]
pub struct ThresholdCurvePointParams {
    #[id = "curve_point_enabled"]
    pub enabled: BoolParam,
    #[id = "curve_point_frequency"]
    pub frequency: FloatParam,
    #[id = "curve_point_offset"]
    pub offset_db: FloatParam,
}

/// The type of threshold to use.
#[derive(Enum, Debug, PartialEq, Eq)]
pub enum ThresholdMode {
    /// Configure the thresholds to offset pink noise. This means that the slope will receive an
    /// additional -3 dB/octave slope.
    #[id = "internal"]
    #[name = "Pink Noise"]
    Internal,
    /// Dynamically reconfigure the thresholds based on a sidechain input. The -3 dB/octave slope
    /// offset is not applied here so the curve stays true to the sidechain input at the default
    /// settings. This works by simply multiplying the sidechain gain levels with the precomputed
    /// threshold, knee start, and knee end values. The sidechain channel linking option determines
    /// how how much of the other channel values to mix in before multiplying the sidechain gain
    /// values with the thresholds.
    #[id = "sidechain"]
    #[name = "Sidechain Matching"]
    SidechainMatch,
    /// Compress the input signal based on the sidechain signal's activity. Can be used to
    /// spectrally duck the input, or to amplify parts of the input based on holes in the sidechain
    /// signal.
    #[id = "sidechain_compress"]
    #[name = "Sidechain Compression"]
    SidechainCompress,
}

impl ThresholdCurvePointParams {
    fn new(index: usize, set_update_thresholds: Arc<dyn Fn(f32) + Send + Sync>) -> Self {
        let set_update_enabled = {
            let set_update_thresholds = set_update_thresholds.clone();
            Arc::new(move |_| set_update_thresholds(0.0))
        };

        Self {
            enabled: BoolParam::new(format!("Curve Point {} Enabled", index + 1), false)
                .with_callback(set_update_enabled)
                .hide()
                .hide_in_generic_ui(),
            frequency: FloatParam::new(
                format!("Curve Point {} Frequency", index + 1),
                1_000.0,
                FloatRange::Skewed {
                    min: THRESHOLD_CURVE_MIN_FREQUENCY_HZ,
                    max: THRESHOLD_CURVE_MAX_FREQUENCY_HZ,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_callback(set_update_thresholds.clone())
            .with_value_to_string(formatters::v2s_f32_hz_then_khz(0))
            .with_string_to_value(formatters::s2v_f32_hz_then_khz())
            .hide()
            .hide_in_generic_ui(),
            offset_db: FloatParam::new(
                format!("Curve Point {} Offset", index + 1),
                0.0,
                FloatRange::Linear {
                    min: -THRESHOLD_CURVE_POINT_OFFSET_LIMIT_DB,
                    max: THRESHOLD_CURVE_POINT_OFFSET_LIMIT_DB,
                },
            )
            .with_callback(set_update_thresholds)
            .with_unit(" dB")
            .with_step_size(0.1)
            .hide()
            .hide_in_generic_ui(),
        }
    }

    pub fn curve_point(&self) -> CurvePoint {
        CurvePoint {
            enabled: self.enabled.value(),
            frequency: self.frequency.value(),
            offset_db: self.offset_db.value(),
        }
    }
}

/// Contains the compressor parameters for both the upwards and downwards compressor banks.
#[derive(Params)]
pub struct CompressorBankParams {
    #[nested(id_prefix = "upwards", group = "upwards")]
    pub upwards: Arc<CompressorParams>,
    #[nested(id_prefix = "downwards", group = "downwards")]
    pub downwards: Arc<CompressorParams>,
}

/// This struct contains the parameters for either the upward or downward compressors. The `Params`
/// trait is implemented manually to avoid copy-pasting parameters for both types of compressor.
/// Both versions will have a parameter ID and a parameter name prefix to distinguish them.
#[derive(Params)]
pub struct CompressorParams {
    /// The compression threshold relative to the target curve.
    #[id = "threshold_offset"]
    pub threshold_offset_db: FloatParam,
    /// The compression ratio. At 1.0 the compressor is disengaged.
    #[id = "ratio"]
    pub ratio: FloatParam,
    /// A `[0, 1]` scaling factor that causes the compressors for the higher registers to have lower
    /// ratios than the compressors for the lower registers. The scaling is applied logarithmically
    /// rather than linearly over the compressors. If this is set to 1.0, then the ratios will be
    /// the same for every compressor. A value of 0.5 means that at
    /// 22,050 Hz, the compression ratio will be 0.5 times that as the one at 0 Hz.
    #[id = "high_freq_rolloff"]
    pub high_freq_ratio_rolloff: FloatParam,
    /// The compression knee width, in decibels.
    #[id = "knee"]
    pub knee_width_db: FloatParam,
}

impl ThresholdParams {
    /// Create a new [`ThresholdParams`] object. Changing any of the threshold parameters causes the
    /// passed compressor bank's thresholds and knee parabolas to be updated.
    pub fn new(compressor_bank: &CompressorBank) -> Self {
        let should_update_downwards_thresholds =
            compressor_bank.should_update_downwards_thresholds.clone();
        let should_update_upwards_thresholds =
            compressor_bank.should_update_upwards_thresholds.clone();
        let should_update_downwards_knee_parabolas = compressor_bank
            .should_update_downwards_knee_parabolas
            .clone();
        let should_update_upwards_knee_parabolas =
            compressor_bank.should_update_upwards_knee_parabolas.clone();
        let set_update_both_thresholds = Arc::new(move |_| {
            should_update_downwards_thresholds.store(true, Ordering::Relaxed);
            should_update_upwards_thresholds.store(true, Ordering::Relaxed);
            should_update_downwards_knee_parabolas.store(true, Ordering::Relaxed);
            should_update_upwards_knee_parabolas.store(true, Ordering::Relaxed);
        });

        ThresholdParams {
            threshold_db: FloatParam::new(
                "Global Threshold",
                -12.0,
                FloatRange::Linear {
                    min: -100.0,
                    max: 20.0,
                },
            )
            .with_callback(set_update_both_thresholds.clone())
            .with_unit(" dB")
            .with_step_size(0.1),
            center_frequency: FloatParam::new(
                "Threshold Center",
                1_000.0,
                FloatRange::Skewed {
                    min: 20.0,
                    max: 20_000.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_callback(set_update_both_thresholds.clone())
            // This includes the unit
            .with_value_to_string(formatters::v2s_f32_hz_then_khz(0))
            .with_string_to_value(formatters::s2v_f32_hz_then_khz()),
            // These are polynomial coefficients that are evaluated in the log/log domain
            // (octaves/decibels). The global threshold is the intercept.
            curve_slope: FloatParam::new(
                "Threshold Slope",
                0.0,
                FloatRange::SymmetricalSkewed {
                    min: -36.0,
                    max: 36.0,
                    factor: FloatRange::skew_factor(-2.0),
                    center: 0.0,
                },
            )
            .with_callback(set_update_both_thresholds.clone())
            .with_value_to_string(Arc::new(|value| {
                let display_value = threshold_slope_to_display_value(value);
                if (display_value * 100.0).round() / 100.0 == 0.0 {
                    String::from("0.00")
                } else {
                    format!("{display_value:.2}")
                }
            }))
            .with_string_to_value(Arc::new(|string| {
                string
                    .trim_end_matches([' ', 'd', 'D', 'b', 'B', '/', 'o', 'O', 'c', 'C', 't', 'T'])
                    .trim()
                    .parse()
                    .ok()
                    .map(threshold_slope_from_display_value)
            }))
            .with_unit(" dB/oct")
            .with_step_size(0.01),
            curve_curve: FloatParam::new(
                "Threshold Curve",
                0.0,
                FloatRange::SymmetricalSkewed {
                    min: -24.0,
                    max: 24.0,
                    factor: FloatRange::skew_factor(-2.0),
                    center: 0.0,
                },
            )
            .with_callback(set_update_both_thresholds.clone())
            .with_unit(" dB/oct²")
            .with_step_size(0.01)
            .hide_in_generic_ui(),
            curve_points: std::array::from_fn(|index| {
                ThresholdCurvePointParams::new(index, set_update_both_thresholds.clone())
            }),

            mode: EnumParam::new("Mode", ThresholdMode::Internal)
                // Not the most efficient way to do this, but it's a bit cleaner than the
                // alternative
                .with_callback(Arc::new(move |_| set_update_both_thresholds(0.0))),
            sc_channel_link: FloatParam::new(
                "SC Channel Link",
                0.8,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_unit("%")
            .with_value_to_string(formatters::v2s_f32_percentage(0))
            .with_string_to_value(formatters::s2v_f32_percentage()),
            gain_smoothing: FloatParam::new(
                "Smoothing",
                0.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_unit("%")
            .with_value_to_string(formatters::v2s_f32_percentage(0))
            .with_string_to_value(formatters::s2v_f32_percentage()),
        }
    }

    /// Build [`CurveParams`] out of this set of parameters.
    pub fn curve_params(&self) -> CurveParams {
        CurveParams {
            intercept: self.threshold_db.value(),
            center_frequency: self.center_frequency.value(),
            // The cheeky 3 additional dB/octave attenuation is to match pink noise with the
            // default settings. When using sidechaining we explicitly don't want this because
            // the curve should be a flat offset to the sidechain input at the default settings.
            slope: match self.mode.value() {
                ThresholdMode::Internal => internal_threshold_slope(self.curve_slope.value()),
                ThresholdMode::SidechainMatch | ThresholdMode::SidechainCompress => {
                    self.curve_slope.value()
                }
            },
            curve: self.curve_curve.value(),
            points: std::array::from_fn(|index| self.curve_points[index].curve_point()),
        }
    }
}

#[inline]
fn internal_threshold_slope(stored_slope: f32) -> f32 {
    stored_slope - PINK_NOISE_SLOPE_OFFSET_DB_PER_OCT
}

#[inline]
fn threshold_slope_to_display_value(stored_slope: f32) -> f32 {
    PINK_NOISE_SLOPE_OFFSET_DB_PER_OCT - stored_slope
}

#[inline]
fn threshold_slope_from_display_value(display_slope: f32) -> f32 {
    PINK_NOISE_SLOPE_OFFSET_DB_PER_OCT - display_slope
}

impl CompressorBankParams {
    /// Create compressor bank parameter objects for both the downwards and upwards compressors of
    /// `compressor`. Changing the ratio, threshold, and knee parameters will cause the compressor
    /// to recompute its values on the next processing cycle.
    pub fn new(compressor: &CompressorBank) -> Self {
        CompressorBankParams {
            downwards: Arc::new(CompressorParams::new(
                DOWNWARDS_NAME_PREFIX,
                compressor.should_update_downwards_thresholds.clone(),
                compressor.should_update_downwards_ratios.clone(),
                compressor.should_update_downwards_knee_parabolas.clone(),
            )),
            upwards: Arc::new(CompressorParams::new(
                UPWARDS_NAME_PREFIX,
                compressor.should_update_upwards_thresholds.clone(),
                compressor.should_update_upwards_ratios.clone(),
                compressor.should_update_upwards_knee_parabolas.clone(),
            )),
        }
    }
}

impl CompressorParams {
    /// Create a new [`CompressorBankParams`] object with a prefix for all parameter names. Changing
    /// any of the threshold, ratio, or knee parameters causes the passed atomics to be updated.
    /// These should be taken from a [`CompressorBank`] so the parameters are linked to it.
    pub fn new(
        name_prefix: &str,
        should_update_thresholds: Arc<AtomicBool>,
        should_update_ratios: Arc<AtomicBool>,
        should_update_knee_parabolas: Arc<AtomicBool>,
    ) -> Self {
        let set_update_thresholds = Arc::new({
            let should_update_knee_parabolas = should_update_knee_parabolas.clone();
            move |_| {
                should_update_thresholds.store(true, Ordering::Relaxed);
                should_update_knee_parabolas.store(true, Ordering::Relaxed);
            }
        });
        let set_update_ratios = Arc::new({
            let should_update_knee_parabolas = should_update_knee_parabolas.clone();
            move |_| {
                should_update_ratios.store(true, Ordering::Relaxed);
                should_update_knee_parabolas.store(true, Ordering::Relaxed);
            }
        });
        let set_update_knee_parabolas = Arc::new(move |_| {
            should_update_knee_parabolas.store(true, Ordering::Relaxed);
        });

        CompressorParams {
            // As explained above, these offsets are relative to the target curve
            threshold_offset_db: FloatParam::new(
                format!("{name_prefix} Offset"),
                0.0,
                FloatRange::Linear {
                    min: -50.0,
                    max: 50.0,
                },
            )
            .with_callback(set_update_thresholds)
            .with_unit(" dB")
            .with_step_size(0.1),
            ratio: FloatParam::new(
                format!("{name_prefix} Ratio"),
                1.0,
                FloatRange::Skewed {
                    min: 1.0,
                    max: 500.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_callback(set_update_ratios.clone())
            .with_step_size(0.01)
            .with_value_to_string(formatters::v2s_compression_ratio(2))
            .with_string_to_value(formatters::s2v_compression_ratio()),
            high_freq_ratio_rolloff: FloatParam::new(
                format!("{name_prefix} Hi-Freq Rolloff"),
                // The upwards bank defaults to a gentle rolloff, while the downwards bank keeps
                // full-band ratios by default.
                if name_prefix == UPWARDS_NAME_PREFIX {
                    0.75
                } else {
                    // When used subtly, no rolloff is usually better for downwards compression
                    0.0
                },
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_callback(set_update_ratios)
            .with_unit("%")
            .with_value_to_string(formatters::v2s_f32_percentage(0))
            .with_string_to_value(formatters::s2v_f32_percentage()),
            knee_width_db: FloatParam::new(
                format!("{name_prefix} Knee"),
                6.0,
                FloatRange::Skewed {
                    min: 0.0,
                    max: 36.0,
                    factor: FloatRange::skew_factor(-1.0),
                },
            )
            .with_callback(set_update_knee_parabolas)
            .with_unit(" dB")
            .with_step_size(0.1),
        }
    }
}

impl CompressorBank {
    /// Set up the compressor for the given channel count and maximum FFT window size. The
    /// compressors won't be initialized yet.
    pub(crate) fn new(
        analyzer_input_data: triple_buffer::Input<AnalyzerData>,
        frozen_ir_input_data: triple_buffer::Input<FrozenIrData>,
        match_curve_runtime: Arc<MatchCurveRuntime>,
        num_channels: usize,
        max_window_size: usize,
    ) -> Self {
        let complex_buffer_len = max_window_size / 2 + 1;

        CompressorBank {
            should_update_downwards_thresholds: Arc::new(AtomicBool::new(true)),
            should_update_upwards_thresholds: Arc::new(AtomicBool::new(true)),
            should_update_downwards_ratios: Arc::new(AtomicBool::new(true)),
            should_update_upwards_ratios: Arc::new(AtomicBool::new(true)),
            should_update_downwards_knee_parabolas: Arc::new(AtomicBool::new(true)),
            should_update_upwards_knee_parabolas: Arc::new(AtomicBool::new(true)),

            ln_freqs: Vec::with_capacity(complex_buffer_len),

            downwards_thresholds_db: Vec::with_capacity(complex_buffer_len),
            downwards_ratios: Vec::with_capacity(complex_buffer_len),
            downwards_knee_parabola_scale: Vec::with_capacity(complex_buffer_len),
            downwards_knee_parabola_intercept: Vec::with_capacity(complex_buffer_len),

            upwards_thresholds_db: Vec::with_capacity(complex_buffer_len),
            upwards_ratios: Vec::with_capacity(complex_buffer_len),
            upwards_knee_parabola_scale: Vec::with_capacity(complex_buffer_len),
            upwards_knee_parabola_intercept: Vec::with_capacity(complex_buffer_len),

            envelopes: vec![Vec::with_capacity(complex_buffer_len); num_channels],
            envelope_followers_timing_scale: 0.0,
            sidechain_spectrum_magnitudes: vec![
                Vec::with_capacity(complex_buffer_len);
                num_channels
            ],
            linked_sidechain_magnitudes: Vec::with_capacity(complex_buffer_len),
            raw_gain_difference_db: Vec::with_capacity(complex_buffer_len),
            smoothed_gain_difference_db: Vec::with_capacity(complex_buffer_len),
            gain_smoothing_prefix_db: Vec::with_capacity(complex_buffer_len + 1),
            frozen_gain_difference_db: vec![Vec::with_capacity(complex_buffer_len); num_channels],
            frozen_gain_snapshot_valid: vec![false; num_channels],
            freeze_was_active: false,
            frozen_ir_snapshot_published: false,
            window_size: 0,
            sample_rate: 1.0,

            analyzer_input_data,
            frozen_ir_input_data,
            analyzer_needs_publish: false,
            match_curve_runtime,
            match_curve_meter: MatchCurveMeter::new(complex_buffer_len),
        }
    }

    /// Change the capacities of the internal buffers to fit new parameters.
    ///
    /// After calling this, callers must invoke [`Self::resize()`] before the next processing pass.
    /// `update_capacity()` only reserves storage and does not make per-bin vectors safe to index
    /// for the active FFT size.
    pub fn update_capacity(&mut self, num_channels: usize, max_window_size: usize) {
        let complex_buffer_len = max_window_size / 2 + 1;

        self.ln_freqs
            .reserve_exact(complex_buffer_len.saturating_sub(self.ln_freqs.len()));

        self.downwards_thresholds_db
            .reserve_exact(complex_buffer_len.saturating_sub(self.downwards_thresholds_db.len()));
        self.downwards_ratios
            .reserve_exact(complex_buffer_len.saturating_sub(self.downwards_ratios.len()));
        self.downwards_knee_parabola_scale.reserve_exact(
            complex_buffer_len.saturating_sub(self.downwards_knee_parabola_scale.len()),
        );
        self.downwards_knee_parabola_intercept.reserve_exact(
            complex_buffer_len.saturating_sub(self.downwards_knee_parabola_intercept.len()),
        );

        self.upwards_thresholds_db
            .reserve_exact(complex_buffer_len.saturating_sub(self.upwards_thresholds_db.len()));
        self.upwards_ratios
            .reserve_exact(complex_buffer_len.saturating_sub(self.upwards_ratios.len()));
        self.upwards_knee_parabola_scale.reserve_exact(
            complex_buffer_len.saturating_sub(self.upwards_knee_parabola_scale.len()),
        );
        self.upwards_knee_parabola_intercept.reserve_exact(
            complex_buffer_len.saturating_sub(self.upwards_knee_parabola_intercept.len()),
        );

        self.envelopes.resize_with(num_channels, Vec::new);
        for envelopes in self.envelopes.iter_mut() {
            envelopes.reserve_exact(complex_buffer_len.saturating_sub(envelopes.len()));
        }

        self.sidechain_spectrum_magnitudes
            .resize_with(num_channels, Vec::new);
        for magnitudes in self.sidechain_spectrum_magnitudes.iter_mut() {
            magnitudes.reserve_exact(complex_buffer_len.saturating_sub(magnitudes.len()));
        }
        self.linked_sidechain_magnitudes.reserve_exact(
            complex_buffer_len.saturating_sub(self.linked_sidechain_magnitudes.len()),
        );
        self.raw_gain_difference_db
            .reserve_exact(complex_buffer_len.saturating_sub(self.raw_gain_difference_db.len()));
        self.smoothed_gain_difference_db.reserve_exact(
            complex_buffer_len.saturating_sub(self.smoothed_gain_difference_db.len()),
        );
        self.gain_smoothing_prefix_db.reserve_exact(
            (complex_buffer_len + 1).saturating_sub(self.gain_smoothing_prefix_db.len()),
        );
        self.match_curve_meter.resize(complex_buffer_len);

        self.frozen_gain_difference_db
            .resize_with(num_channels, Vec::new);
        for gains in self.frozen_gain_difference_db.iter_mut() {
            gains.reserve_exact(complex_buffer_len.saturating_sub(gains.len()));
        }
        self.frozen_gain_snapshot_valid.resize(num_channels, false);
    }

    /// Resize the number of compressors to match the current window size. Also precomputes the
    /// 2-log frequencies for each bin.
    ///
    /// If the window size is larger than the maximum window size, then this will allocate.
    pub fn resize(&mut self, buffer_config: &BufferConfig, window_size: usize) {
        let complex_buffer_len = window_size / 2 + 1;

        // These 2-log frequencies are needed when updating the compressor parameters, so we'll just
        // precompute them to avoid having to repeat the same expensive computations all the time
        self.ln_freqs.resize(complex_buffer_len, 0.0);
        // The first one should always stay at zero, `0.0f32.ln() == NaN`.
        for (i, ln_freq) in self.ln_freqs.iter_mut().enumerate().skip(1) {
            let freq = (i as f32 / window_size as f32) * buffer_config.sample_rate;
            *ln_freq = freq.ln();
        }

        self.downwards_thresholds_db.resize(complex_buffer_len, 1.0);
        self.downwards_ratios.resize(complex_buffer_len, 1.0);
        self.downwards_knee_parabola_scale
            .resize(complex_buffer_len, 1.0);
        self.downwards_knee_parabola_intercept
            .resize(complex_buffer_len, 1.0);

        self.upwards_thresholds_db.resize(complex_buffer_len, 1.0);
        self.upwards_ratios.resize(complex_buffer_len, 1.0);
        self.upwards_knee_parabola_scale
            .resize(complex_buffer_len, 1.0);
        self.upwards_knee_parabola_intercept
            .resize(complex_buffer_len, 1.0);

        for envelopes in self.envelopes.iter_mut() {
            envelopes.resize(complex_buffer_len, ENVELOPE_INIT_VALUE);
        }

        for magnitudes in self.sidechain_spectrum_magnitudes.iter_mut() {
            magnitudes.resize(complex_buffer_len, 0.0);
        }
        self.linked_sidechain_magnitudes
            .resize(complex_buffer_len, 0.0);
        self.raw_gain_difference_db.resize(complex_buffer_len, 0.0);
        self.smoothed_gain_difference_db
            .resize(complex_buffer_len, 0.0);
        self.gain_smoothing_prefix_db
            .resize(complex_buffer_len + 1, 0.0);
        self.match_curve_meter.resize(complex_buffer_len);

        for gains in self.frozen_gain_difference_db.iter_mut() {
            gains.resize(complex_buffer_len, 0.0);
            gains.fill(0.0);
        }
        self.frozen_gain_snapshot_valid.fill(false);
        self.freeze_was_active = false;
        self.frozen_ir_snapshot_published = false;

        self.window_size = window_size;
        self.sample_rate = buffer_config.sample_rate;
        self.publish_frozen_ir_unavailable();

        // The compressors need to be updated on the next processing cycle
        self.should_update_downwards_thresholds
            .store(true, Ordering::Relaxed);
        self.should_update_upwards_thresholds
            .store(true, Ordering::Relaxed);
        self.should_update_downwards_ratios
            .store(true, Ordering::Relaxed);
        self.should_update_upwards_ratios
            .store(true, Ordering::Relaxed);
        self.should_update_downwards_knee_parabolas
            .store(true, Ordering::Relaxed);
        self.should_update_upwards_knee_parabolas
            .store(true, Ordering::Relaxed);
    }

    /// Get the sample rate this compressor bank was configured for.
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Clear out the envelope followers.
    pub fn reset(&mut self) {
        // This will make the timings instant for the first iteration after a reset and then slowly
        // fade the timings back to their intended values so the envelope followers can settle in.
        // Otherwise suspending and resetting the plugin, or changing the window size, may result in
        // some huge spikes.
        self.envelope_followers_timing_scale = 0.0;
        self.clear_frozen_gain_snapshots(true);

        // Sidechain data doesn't need to be reset as it will be overwritten immediately before use
    }

    fn publish_frozen_ir_snapshot(&mut self, overlap_times: usize) {
        let snapshot = self.frozen_ir_input_data.input_buffer();
        snapshot.valid = true;
        snapshot.sample_rate = self.sample_rate;
        snapshot.window_size = self.window_size;
        snapshot.overlap_times = overlap_times;
        debug_assert!(self.frozen_gain_difference_db.len() <= 2);
        snapshot.num_channels = self.frozen_gain_difference_db.len().min(2);

        let num_bins = self.window_size / 2 + 1;
        for channel_idx in 0..snapshot.gain_difference_db.len() {
            if channel_idx < snapshot.num_channels {
                snapshot.gain_difference_db[channel_idx][..num_bins]
                    .copy_from_slice(&self.frozen_gain_difference_db[channel_idx][..num_bins]);
            } else {
                snapshot.gain_difference_db[channel_idx][..num_bins].fill(0.0);
            }
            snapshot.gain_difference_db[channel_idx][num_bins..].fill(0.0);
        }

        self.frozen_ir_input_data.publish();
    }

    fn publish_frozen_ir_unavailable(&mut self) {
        let snapshot = self.frozen_ir_input_data.input_buffer();
        snapshot.valid = false;
        snapshot.sample_rate = self.sample_rate;
        snapshot.window_size = self.window_size;
        snapshot.overlap_times = 0;
        debug_assert!(self.frozen_gain_difference_db.len() <= 2);
        snapshot.num_channels = self.frozen_gain_difference_db.len().min(2);
        for channel in &mut snapshot.gain_difference_db {
            channel.fill(0.0);
        }

        self.frozen_ir_input_data.publish();
    }

    /// Apply the magnitude compression to a buffer of FFT bins. The compressors are first updated
    /// if needed. The overlap amount is needed to compute the effective sample rate. The
    /// `first_non_dc_bin` argument is used to avoid upwards compression on the DC bins, or the
    /// neighbouring bins the DC signal may have been convolved into because of the Hann window
    /// function.
    pub fn process(
        &mut self,
        buffer: &mut [Complex32],
        channel_idx: usize,
        params: &SpectralCompressorParams,
        overlap_times: usize,
        first_non_dc_bin: usize,
    ) {
        assert_eq!(buffer.len(), self.ln_freqs.len());
        let mode = params.threshold.mode.value();
        let freeze_enabled = params.global.compressor_freeze.value();
        let freeze_supported = matches!(
            mode,
            ThresholdMode::Internal
                | ThresholdMode::SidechainMatch
                | ThresholdMode::SidechainCompress
        );
        self.sync_freeze_state(freeze_enabled && freeze_supported);

        // The gain difference/reduction amounts are accumulated in `self.analyzer_input_data`. When
        // processing the last channel, this data is divided by the channel count, the envelope
        // follower data is added, and the data is then sent to the editor so it can be displayed.
        // `analyzer_input_data` contains excess capacity so it can handle any supported window
        // size, so all operations on it are limited to the actual number of used bins.
        let num_bins = buffer.len();
        let num_channels = self.sidechain_spectrum_magnitudes.len();
        let should_update_analyzer_data = params.editor_state.is_open();
        if should_update_analyzer_data && channel_idx == 0 {
            // NOTE: This may briefly show a huge amount of accumulated data when the editor has
            //       just been opened. If this doesn't look too obvious or too jarring this is
            //       probably worth letting it be like this.
            let analyzer_input_data = self.analyzer_input_data.input_buffer();
            analyzer_input_data.gain_difference_db[..num_bins].fill(0.0);
        }

        self.update_if_needed(params);
        if self.match_curve_runtime.begin_running_if_requested() {
            self.match_curve_meter.start(self.sample_rate);
        }

        match mode {
            ThresholdMode::Internal => {
                self.update_envelopes(buffer, channel_idx, params, overlap_times);
                self.measure_match_curve_complex(buffer, channel_idx, params, overlap_times);
                self.compress(
                    buffer,
                    channel_idx,
                    params,
                    first_non_dc_bin,
                    freeze_enabled,
                    should_update_analyzer_data,
                )
            }
            ThresholdMode::SidechainMatch => {
                self.update_envelopes(buffer, channel_idx, params, overlap_times);
                self.measure_match_curve_complex(buffer, channel_idx, params, overlap_times);
                self.compress_sidechain_match(
                    buffer,
                    channel_idx,
                    params,
                    first_non_dc_bin,
                    freeze_enabled,
                    should_update_analyzer_data,
                )
            }
            ThresholdMode::SidechainCompress => {
                // This mode uses regular compression, but the envelopes are computed from the
                // sidechain input magnitudes. These are already set in `process_sidechain`. This
                // separate envelope updating function is needed for the channel linking.
                self.update_envelopes_sidechain(channel_idx, params, overlap_times);
                self.measure_match_curve_sidechain(channel_idx, params, overlap_times);
                self.compress(
                    buffer,
                    channel_idx,
                    params,
                    first_non_dc_bin,
                    freeze_enabled,
                    should_update_analyzer_data,
                )
            }
        };

        if freeze_enabled
            && freeze_supported
            && !self.frozen_ir_snapshot_published
            && self.frozen_gain_snapshot_valid.iter().all(|valid| *valid)
        {
            self.publish_frozen_ir_snapshot(overlap_times);
            self.frozen_ir_snapshot_published = true;
        }

        // When processing the last channel we can finalize the spectrum analyzer data and send it
        // to the editor for display
        if should_update_analyzer_data && channel_idx == num_channels - 1 {
            let analyzer_input_data = self.analyzer_input_data.input_buffer();

            // The editor needs to know about this too so it can draw the spectra correctly
            analyzer_input_data.curve_params = params.threshold.curve_params();
            analyzer_input_data.curve_offsets_db = (
                params.compressors.upwards.threshold_offset_db.value(),
                params.compressors.downwards.threshold_offset_db.value(),
            );
            analyzer_input_data.num_bins = num_bins;

            // The gain reduction data needs to be averaged, see above
            let channel_multiplier = (num_channels as f32).recip();
            for gain_difference_db in &mut analyzer_input_data.gain_difference_db[..num_bins] {
                *gain_difference_db *= channel_multiplier;
            }

            // The spectrum analyzer data has not yet been added
            assert!(self.envelopes.len() == num_channels);
            assert!(self.envelopes[0].len() >= num_bins);
            for (bin_idx, spectrum_data) in analyzer_input_data.envelope_followers[..num_bins]
                .iter_mut()
                .enumerate()
            {
                *spectrum_data = 0.0;
                for channel_idx in 0..num_channels {
                    *spectrum_data += self.envelopes[channel_idx][bin_idx];
                }

                *spectrum_data *= channel_multiplier;
            }

            // After filling the object with data it can be sent to the editor. This happens
            // automatically when using the `.write()` interface, but since `AnalyzerData` contains
            // a lot of padding and we only use the first `num_bins` of the arrays that would be a
            // bit wasteful.
            // NOTE: The actual publish is deferred until after the STFT callback has finished
            //       with the compressor bank.
            self.analyzer_needs_publish = true;
        }
    }

    /// Publish analyzer data prepared during the compressor processing pass.
    pub fn publish_analyzer_if_needed(&mut self) {
        if self.analyzer_needs_publish {
            self.analyzer_input_data.publish();
            self.analyzer_needs_publish = false;
        }
    }

    /// Set the sidechain frequency spectrum magnitudes just before a [`process()`][Self::process()]
    /// call. These will be multiplied with the existing compressor thresholds and knee values to
    /// get the effective values for use with sidechaining.
    pub fn process_sidechain(&mut self, sc_buffer: &[Complex32], channel_idx: usize) {
        nih_debug_assert_eq!(sc_buffer.len(), self.ln_freqs.len());

        self.update_sidechain_spectra(sc_buffer, channel_idx);
    }

    /// Update the envelope followers based on the bin magnitudes.
    fn update_envelopes(
        &mut self,
        buffer: &[Complex32],
        channel_idx: usize,
        params: &SpectralCompressorParams,
        overlap_times: usize,
    ) {
        let effective_sample_rate =
            self.sample_rate / (self.window_size as f32 / overlap_times as f32);

        // The timings are scaled by `self.envelope_followers_timing_scale` to allow the envelope
        // followers to settle in quicker after a reset
        let attack_ms =
            params.global.compressor_attack_ms.value() * self.envelope_followers_timing_scale;
        let release_ms =
            params.global.compressor_release_ms.value() * self.envelope_followers_timing_scale;

        // This needs to gradually fade from 0.0 back to 1.0 after a reset
        if self.envelope_followers_timing_scale < 1.0 && channel_idx == self.envelopes.len() - 1 {
            let delta =
                ((ENVELOPE_FOLLOWER_TIMING_FADE_MS / 1000.0) * effective_sample_rate).recip();
            self.envelope_followers_timing_scale =
                (self.envelope_followers_timing_scale + delta).min(1.0);
        }

        // The coefficient the old envelope value is multiplied by when the current rectified sample
        // value is above the envelope's value. The 0 to 1 step response retains 36.8% of the old
        // value after the attack time has elapsed, and current value is 63.2% of the way towards 1.
        // The effective sample rate needs to compensate for the periodic nature of the STFT
        // operation. Since with a 2048 sample window and 4x overlap, you'd run this function once
        // for every 512 samples.
        let attack_old_t = if attack_ms == 0.0 {
            0.0
        } else {
            (-1.0 / (attack_ms / 1000.0 * effective_sample_rate)).exp()
        };
        let attack_new_t = 1.0 - attack_old_t;
        // The same as `attack_old_t`, but for the release phase of the envelope follower
        let release_old_t = if release_ms == 0.0 {
            0.0
        } else {
            (-1.0 / (release_ms / 1000.0 * effective_sample_rate)).exp()
        };
        let release_new_t = 1.0 - release_old_t;

        for (bin, envelope) in buffer.iter().zip(self.envelopes[channel_idx].iter_mut()) {
            let magnitude = bin.norm();
            if *envelope > magnitude {
                // Release stage
                *envelope = (release_old_t * *envelope) + (release_new_t * magnitude);
            } else {
                // Attack stage
                *envelope = (attack_old_t * *envelope) + (attack_new_t * magnitude);
            }
        }
    }

    /// The same as [`update_envelopes()`][Self::update_envelopes()], but based on the previously
    /// set sidechain bin magnitudes. This allows for channel linking.
    /// [`process_sidechain()`][Self::process_sidechain()] needs to be called for all channels
    /// before this function can be used to set the magnitude spectra.
    fn update_envelopes_sidechain(
        &mut self,
        channel_idx: usize,
        params: &SpectralCompressorParams,
        overlap_times: usize,
    ) {
        let effective_sample_rate =
            self.sample_rate / (self.window_size as f32 / overlap_times as f32);

        // The timings are scaled by `self.envelope_followers_timing_scale` to allow the envelope
        // followers to settle in quicker after a reset
        let attack_ms =
            params.global.compressor_attack_ms.value() * self.envelope_followers_timing_scale;
        let release_ms =
            params.global.compressor_release_ms.value() * self.envelope_followers_timing_scale;

        // This needs to gradually fade from 0.0 back to 1.0 after a reset
        if self.envelope_followers_timing_scale < 1.0 && channel_idx == self.envelopes.len() - 1 {
            let delta =
                ((ENVELOPE_FOLLOWER_TIMING_FADE_MS / 1000.0) * effective_sample_rate).recip();
            self.envelope_followers_timing_scale =
                (self.envelope_followers_timing_scale + delta).min(1.0);
        }

        // See `update_envelopes()`
        let attack_old_t = if attack_ms == 0.0 {
            0.0
        } else {
            (-1.0 / (attack_ms / 1000.0 * effective_sample_rate)).exp()
        };
        let attack_new_t = 1.0 - attack_old_t;
        let release_old_t = if release_ms == 0.0 {
            0.0
        } else {
            (-1.0 / (release_ms / 1000.0 * effective_sample_rate)).exp()
        };
        let release_new_t = 1.0 - release_old_t;

        // For the channel linking
        let num_channels = self.sidechain_spectrum_magnitudes.len() as f32;
        let other_channels_t = params.threshold.sc_channel_link.value() / num_channels;
        let this_channel_t = 1.0 - (other_channels_t * (num_channels - 1.0));

        for (bin_idx, envelope) in self.envelopes[channel_idx].iter_mut().enumerate() {
            // In this mode the envelopes are set based on the sidechain signal, taking channel
            // linking into account
            let sidechain_magnitude: f32 = self
                .sidechain_spectrum_magnitudes
                .iter()
                .enumerate()
                .map(|(sidechain_channel_idx, magnitudes)| {
                    let t = if sidechain_channel_idx == channel_idx {
                        this_channel_t
                    } else {
                        other_channels_t
                    };

                    magnitudes[bin_idx] * t
                })
                .sum::<f32>();

            if *envelope > sidechain_magnitude {
                // Release stage
                *envelope = (release_old_t * *envelope) + (release_new_t * sidechain_magnitude);
            } else {
                // Attack stage
                *envelope = (attack_old_t * *envelope) + (attack_new_t * sidechain_magnitude);
            }
        }
    }

    /// Update the spectral data using the sidechain input
    fn update_sidechain_spectra(&mut self, sc_buffer: &[Complex32], channel_idx: usize) {
        nih_debug_assert!(channel_idx < self.sidechain_spectrum_magnitudes.len());

        for (bin, magnitude) in sc_buffer
            .iter()
            .zip(self.sidechain_spectrum_magnitudes[channel_idx].iter_mut())
        {
            *magnitude = bin.norm();
        }
    }

    fn measure_match_curve_complex(
        &mut self,
        buffer: &[Complex32],
        channel_idx: usize,
        params: &SpectralCompressorParams,
        overlap_times: usize,
    ) {
        if !self.match_curve_meter.is_active() {
            return;
        }

        self.match_curve_meter
            .measure_magnitudes(buffer.iter().map(|bin| bin.norm()));
        self.finish_match_curve_channel_if_needed(channel_idx, params, overlap_times);
    }

    fn measure_match_curve_sidechain(
        &mut self,
        channel_idx: usize,
        params: &SpectralCompressorParams,
        overlap_times: usize,
    ) {
        if !self.match_curve_meter.is_active() {
            return;
        }

        let num_channels = self.sidechain_spectrum_magnitudes.len() as f32;
        let other_channels_t = params.threshold.sc_channel_link.value() / num_channels;
        let this_channel_t = 1.0 - (other_channels_t * (num_channels - 1.0));

        self.match_curve_meter
            .measure_magnitudes((0..self.ln_freqs.len()).map(|bin_idx| {
                self.sidechain_spectrum_magnitudes
                    .iter()
                    .enumerate()
                    .map(|(sidechain_channel_idx, magnitudes)| {
                        let t = if sidechain_channel_idx == channel_idx {
                            this_channel_t
                        } else {
                            other_channels_t
                        };

                        magnitudes[bin_idx] * t
                    })
                    .sum::<f32>()
            }));
        self.finish_match_curve_channel_if_needed(channel_idx, params, overlap_times);
    }

    fn finish_match_curve_channel_if_needed(
        &mut self,
        channel_idx: usize,
        params: &SpectralCompressorParams,
        overlap_times: usize,
    ) {
        if channel_idx != self.sidechain_spectrum_magnitudes.len() - 1 {
            return;
        }

        let hop_frames = (self.window_size / overlap_times).max(1);
        let threshold = &params.threshold;
        let fixed_slope = match threshold.mode.value() {
            ThresholdMode::Internal => internal_threshold_slope(threshold.curve_slope.value()),
            ThresholdMode::SidechainMatch | ThresholdMode::SidechainCompress => {
                threshold.curve_slope.value()
            }
        };
        if let Some(result) = self.match_curve_meter.advance_and_finish(
            hop_frames,
            &self.ln_freqs,
            params.compressors.downwards.threshold_offset_db.value(),
            threshold.center_frequency.value(),
            fixed_slope,
        ) {
            self.match_curve_runtime.publish_result(result);
        }
    }

    /// Actually do the thing. [`Self::update_envelopes()`] or
    /// [`Self::update_envelopes_sidechain()`] must have been called before calling this.
    ///
    /// # Panics
    ///
    /// Panics if the buffer does not have the same length as the one that was passed to the last
    /// `resize()` call.
    fn compress(
        &mut self,
        buffer: &mut [Complex32],
        channel_idx: usize,
        params: &SpectralCompressorParams,
        first_non_dc_bin: usize,
        freeze_enabled: bool,
        should_update_analyzer_data: bool,
    ) {
        let downwards_knee_width_db = params.compressors.downwards.knee_width_db.value();
        let upwards_knee_width_db = params.compressors.upwards.knee_width_db.value();

        assert!(self.downwards_thresholds_db.len() == buffer.len());
        assert!(self.downwards_ratios.len() == buffer.len());
        assert!(self.downwards_knee_parabola_scale.len() == buffer.len());
        assert!(self.downwards_knee_parabola_intercept.len() == buffer.len());
        assert!(self.upwards_thresholds_db.len() == buffer.len());
        assert!(self.upwards_ratios.len() == buffer.len());
        assert!(self.upwards_knee_parabola_scale.len() == buffer.len());
        assert!(self.upwards_knee_parabola_intercept.len() == buffer.len());
        assert!(self.frozen_gain_difference_db[channel_idx].len() == buffer.len());
        assert!(self.raw_gain_difference_db.len() == buffer.len());
        assert!(self.smoothed_gain_difference_db.len() == buffer.len());
        // NOTE: In the sidechain compression mode these envelopes are computed from the sidechain
        //       signal instead of the main input
        for (bin_idx, envelope) in self.envelopes[channel_idx].iter().enumerate() {
            // We'll apply the transfer curve to the envelope signal, and then scale the complex
            // `bin` by the gain difference
            let envelope_db = util::gain_to_db_fast_epsilon(*envelope);

            let downwards_threshold_db = &self.downwards_thresholds_db[bin_idx];
            let downwards_ratio = &self.downwards_ratios[bin_idx];
            let downwards_knee_parabola_scale = &self.downwards_knee_parabola_scale[bin_idx];
            let downwards_knee_parabola_intercept =
                &self.downwards_knee_parabola_intercept[bin_idx];
            let downwards_compressed = compress_downwards(
                envelope_db,
                *downwards_threshold_db,
                *downwards_ratio,
                downwards_knee_width_db,
                *downwards_knee_parabola_scale,
                *downwards_knee_parabola_intercept,
            );

            // Upwards compression should not happen when the signal is _too_ quiet as we'd only be
            // amplifying noise. We also don't want to amplify DC noise and super low frequencies.
            let upwards_threshold_db = &self.upwards_thresholds_db[bin_idx];
            let upwards_ratio = &self.upwards_ratios[bin_idx];
            let upwards_knee_parabola_scale = &self.upwards_knee_parabola_scale[bin_idx];
            let upwards_knee_parabola_intercept = &self.upwards_knee_parabola_intercept[bin_idx];
            let upwards_compressed = if bin_idx >= first_non_dc_bin
                && *upwards_ratio != 1.0
                && envelope_db > util::MINUS_INFINITY_DB
            {
                compress_upwards(
                    envelope_db,
                    *upwards_threshold_db,
                    *upwards_ratio,
                    upwards_knee_width_db,
                    *upwards_knee_parabola_scale,
                    *upwards_knee_parabola_intercept,
                )
            } else {
                envelope_db
            };

            // If the compressed output is -10 dBFS and the envelope follower was at -6 dBFS, then we
            // want to apply -4 dB of gain to the bin
            self.raw_gain_difference_db[bin_idx] =
                downwards_compressed + upwards_compressed - (envelope_db * 2.0);
        }

        self.smooth_gain_differences(
            params.threshold.gain_smoothing.value(),
            first_non_dc_bin,
            buffer.len(),
        );
        self.apply_gain_differences(
            buffer,
            channel_idx,
            freeze_enabled,
            should_update_analyzer_data,
        );
    }

    /// The same as [`compress()`][Self::compress()], but multiplying the threshold and knee values
    /// with the sidechain gains.
    ///
    /// # Panics
    ///
    /// Panics if the buffer does not have the same length as the one that was passed to the last
    /// `resize()` call.
    fn compress_sidechain_match(
        &mut self,
        buffer: &mut [Complex32],
        channel_idx: usize,
        params: &SpectralCompressorParams,
        first_non_dc_bin: usize,
        freeze_enabled: bool,
        should_update_analyzer_data: bool,
    ) {
        let downwards_knee_width_db = params.compressors.downwards.knee_width_db.value();
        let upwards_knee_width_db = params.compressors.upwards.knee_width_db.value();

        // For the channel linking
        let num_channels = self.sidechain_spectrum_magnitudes.len() as f32;
        let other_channels_t = params.threshold.sc_channel_link.value() / num_channels;
        let this_channel_t = 1.0 - (other_channels_t * (num_channels - 1.0));

        assert!(self.sidechain_spectrum_magnitudes[channel_idx].len() == buffer.len());
        assert!(self.downwards_thresholds_db.len() == buffer.len());
        assert!(self.downwards_ratios.len() == buffer.len());
        assert!(self.upwards_thresholds_db.len() == buffer.len());
        assert!(self.upwards_ratios.len() == buffer.len());
        assert!(self.frozen_gain_difference_db[channel_idx].len() == buffer.len());
        assert!(self.linked_sidechain_magnitudes.len() == buffer.len());
        assert!(self.raw_gain_difference_db.len() == buffer.len());
        assert!(self.smoothed_gain_difference_db.len() == buffer.len());

        for (bin_idx, linked_magnitude) in self.linked_sidechain_magnitudes.iter_mut().enumerate() {
            *linked_magnitude = self
                .sidechain_spectrum_magnitudes
                .iter()
                .enumerate()
                .map(|(sidechain_channel_idx, magnitudes)| {
                    let t = if sidechain_channel_idx == channel_idx {
                        this_channel_t
                    } else {
                        other_channels_t
                    };

                    magnitudes[bin_idx] * t
                })
                .sum::<f32>()
                // The thresholds may never reach zero as they are used in divisions.
                .max(f32::EPSILON);
        }

        for (bin_idx, envelope) in self.envelopes[channel_idx].iter().enumerate() {
            let envelope_db = util::gain_to_db_fast_epsilon(*envelope);

            // The idea here is that we scale the compressor thresholds/knee values by the sidechain
            // signal, thus sort of creating a dynamic multiband compressor
            let sidechain_scale = self.linked_sidechain_magnitudes[bin_idx];
            let sidechain_scale_db = util::gain_to_db_fast_epsilon(sidechain_scale);

            // Notice how the threshold and knee values are scaled here
            let downwards_threshold_db = (self.downwards_thresholds_db[bin_idx]
                + sidechain_scale_db)
                .max(util::MINUS_INFINITY_DB);
            let downwards_ratio = &self.downwards_ratios[bin_idx];
            // Because the thresholds are scaled based on the sidechain input, we also need to
            // recompute the knee coefficients
            let (downwards_knee_parabola_scale, downwards_knee_parabola_intercept) =
                downwards_soft_knee_coefficients(
                    downwards_threshold_db,
                    downwards_knee_width_db,
                    *downwards_ratio,
                );
            let downwards_compressed = compress_downwards(
                envelope_db,
                downwards_threshold_db,
                *downwards_ratio,
                downwards_knee_width_db,
                downwards_knee_parabola_scale,
                downwards_knee_parabola_intercept,
            );

            let upwards_threshold_db = (self.upwards_thresholds_db[bin_idx] + sidechain_scale_db)
                .max(util::MINUS_INFINITY_DB);
            let upwards_ratio = &self.upwards_ratios[bin_idx];
            let upwards_compressed = if bin_idx >= first_non_dc_bin
                && *upwards_ratio != 1.0
                && envelope_db > util::MINUS_INFINITY_DB
            {
                let (upwards_knee_parabola_scale, upwards_knee_parabola_intercept) =
                    upwards_soft_knee_coefficients(
                        upwards_threshold_db,
                        upwards_knee_width_db,
                        *upwards_ratio,
                    );
                compress_upwards(
                    envelope_db,
                    upwards_threshold_db,
                    *upwards_ratio,
                    upwards_knee_width_db,
                    upwards_knee_parabola_scale,
                    upwards_knee_parabola_intercept,
                )
            } else {
                envelope_db
            };

            // If the comprssed output is -10 dBFS and the envelope follower was at -6 dBFS, then we
            // want to apply -4 dB of gain to the bin
            self.raw_gain_difference_db[bin_idx] =
                downwards_compressed + upwards_compressed - (envelope_db * 2.0);
        }

        self.smooth_gain_differences(
            params.threshold.gain_smoothing.value(),
            first_non_dc_bin,
            buffer.len(),
        );
        self.apply_gain_differences(
            buffer,
            channel_idx,
            freeze_enabled,
            should_update_analyzer_data,
        );
    }

    fn smooth_gain_differences(
        &mut self,
        smoothing_amount: f32,
        first_non_dc_bin: usize,
        num_bins: usize,
    ) {
        assert!(self.raw_gain_difference_db.len() >= num_bins);
        assert!(self.smoothed_gain_difference_db.len() >= num_bins);
        assert!(self.gain_smoothing_prefix_db.len() > num_bins);
        assert!(self.ln_freqs.len() >= num_bins);

        let smoothing_amount = smoothing_amount.clamp(0.0, 1.0);
        if smoothing_amount <= f32::EPSILON || num_bins <= 2 {
            self.smoothed_gain_difference_db[..num_bins]
                .copy_from_slice(&self.raw_gain_difference_db[..num_bins]);
            return;
        }

        self.gain_smoothing_prefix_db[0] = 0.0;
        for bin_idx in 0..num_bins {
            self.gain_smoothing_prefix_db[bin_idx + 1] =
                self.gain_smoothing_prefix_db[bin_idx] + self.raw_gain_difference_db[bin_idx];
        }

        self.smoothed_gain_difference_db[0] = self.raw_gain_difference_db[0];

        let radius_ln = smoothing_amount * GAIN_SMOOTHING_MAX_RADIUS_LN;
        let mut left_idx = 1;
        let mut right_idx = 1;
        for bin_idx in 1..num_bins {
            let center_ln = self.ln_freqs[bin_idx];

            while left_idx < bin_idx && center_ln - self.ln_freqs[left_idx] > radius_ln {
                left_idx += 1;
            }

            right_idx = right_idx.max(bin_idx);
            while right_idx + 1 < num_bins && self.ln_freqs[right_idx + 1] - center_ln <= radius_ln
            {
                right_idx += 1;
            }

            let sum = self.gain_smoothing_prefix_db[right_idx + 1]
                - self.gain_smoothing_prefix_db[left_idx];
            let count = (right_idx - left_idx + 1) as f32;
            let averaged = sum / count;
            let raw = self.raw_gain_difference_db[bin_idx];
            let mut smoothed = raw + ((averaged - raw) * smoothing_amount);

            if bin_idx < first_non_dc_bin && smoothed > 0.0 {
                smoothed = 0.0;
            }

            self.smoothed_gain_difference_db[bin_idx] = smoothed;
        }
    }

    fn apply_gain_differences(
        &mut self,
        buffer: &mut [Complex32],
        channel_idx: usize,
        freeze_enabled: bool,
        should_update_analyzer_data: bool,
    ) {
        assert!(self.frozen_gain_difference_db[channel_idx].len() == buffer.len());
        assert!(self.smoothed_gain_difference_db.len() >= buffer.len());

        let should_capture_snapshot =
            freeze_enabled && !self.frozen_gain_snapshot_valid[channel_idx];

        {
            // The gain reduction values are always added to the arrays stored in this object. This
            // makes it possible to visualize the gain reduction without a lot of conditionals.
            let analyzer_input_data = self.analyzer_input_data.input_buffer();
            assert!(analyzer_input_data.gain_difference_db.len() >= buffer.len());
            let frozen_gain_difference_db = &mut self.frozen_gain_difference_db[channel_idx];
            let smoothed_gain_difference_db = &self.smoothed_gain_difference_db[..buffer.len()];

            for (bin_idx, bin) in buffer.iter_mut().enumerate() {
                let smoothed_gain_difference_db = smoothed_gain_difference_db[bin_idx];
                let applied_gain_difference_db = if freeze_enabled {
                    if should_capture_snapshot {
                        frozen_gain_difference_db[bin_idx] = smoothed_gain_difference_db;
                    }

                    frozen_gain_difference_db[bin_idx]
                } else {
                    smoothed_gain_difference_db
                };
                if should_update_analyzer_data {
                    analyzer_input_data.gain_difference_db[bin_idx] += applied_gain_difference_db;
                }

                *bin *= util::db_to_gain_fast(applied_gain_difference_db);
            }
        }

        if should_capture_snapshot {
            self.frozen_gain_snapshot_valid[channel_idx] = true;
        }
    }

    fn sync_freeze_state(&mut self, freeze_enabled: bool) {
        if freeze_enabled {
            if !self.freeze_was_active {
                self.clear_frozen_gain_snapshots(true);
            }
        } else if self.freeze_was_active
            || self.frozen_gain_snapshot_valid.iter().any(|valid| *valid)
        {
            self.clear_frozen_gain_snapshots(false);
        }
    }

    fn clear_frozen_gain_snapshots(&mut self, freeze_active: bool) {
        for gains in &mut self.frozen_gain_difference_db {
            gains.fill(0.0);
        }
        self.frozen_gain_snapshot_valid.fill(false);
        self.freeze_was_active = freeze_active;
        self.frozen_ir_snapshot_published = false;
        self.publish_frozen_ir_unavailable();
    }

    /// Update the compressors if needed. This is called just before processing, and the compressors
    /// are updated in accordance to the atomic flags set on this struct.
    fn update_if_needed(&mut self, params: &SpectralCompressorParams) {
        // The threshold curve is a polynomial in log-log (decibels-octaves) space
        let curve_params = params.threshold.curve_params();
        let curve = Curve::new(&curve_params);

        if self
            .should_update_downwards_thresholds
            .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let downwards_intercept = params.compressors.downwards.threshold_offset_db.value();
            for (ln_freq, threshold_db) in self
                .ln_freqs
                .iter()
                .zip(self.downwards_thresholds_db.iter_mut())
            {
                *threshold_db = curve.evaluate_ln(*ln_freq) + downwards_intercept;
            }
        }

        if self
            .should_update_upwards_thresholds
            .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let upwards_intercept = params.compressors.upwards.threshold_offset_db.value();
            for (ln_freq, threshold_db) in self
                .ln_freqs
                .iter()
                .zip(self.upwards_thresholds_db.iter_mut())
            {
                *threshold_db = curve.evaluate_ln(*ln_freq) + upwards_intercept;
            }
        }

        if self
            .should_update_downwards_ratios
            .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            // If the high-frequency rolloff is enabled then higher frequency bins will have their
            // ratios reduced to reduce harshness. This follows the octave scale. It's easier to do
            // this cleanly using reciprocals.
            let target_ratio_recip = params.compressors.downwards.ratio.value().recip();
            let downwards_high_freq_ratio_rolloff =
                params.compressors.downwards.high_freq_ratio_rolloff.value();
            for (ln_freq, ratio) in self.ln_freqs.iter().zip(self.downwards_ratios.iter_mut()) {
                // Clamp to avoid negative low-frequency values in edge cases.
                let octave_fraction = (ln_freq / HIGH_FREQ_RATIO_ROLLOFF_FREQUENCY_LN).max(0.0);
                let rolloff_t = octave_fraction * downwards_high_freq_ratio_rolloff;

                // If the octave fraction times the rolloff amount is high, then this should get
                // closer to `high_freq_ratio_rolloff` (which is in [0, 1]).
                let ratio_recip = (target_ratio_recip * (1.0 - rolloff_t)) + rolloff_t;
                *ratio = ratio_recip.recip();
            }
        }

        if self
            .should_update_upwards_ratios
            .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let target_ratio_recip = params.compressors.upwards.ratio.value().recip();
            let upwards_high_freq_ratio_rolloff =
                params.compressors.upwards.high_freq_ratio_rolloff.value();
            for (ln_freq, ratio) in self.ln_freqs.iter().zip(self.upwards_ratios.iter_mut()) {
                // Clamp to avoid negative low-frequency values in edge cases.
                let octave_fraction = (ln_freq / HIGH_FREQ_RATIO_ROLLOFF_FREQUENCY_LN).max(0.0);
                let rolloff_t = octave_fraction * upwards_high_freq_ratio_rolloff;

                let ratio_recip = (target_ratio_recip * (1.0 - rolloff_t)) + rolloff_t;
                *ratio = ratio_recip.recip();
            }
        }

        if self
            .should_update_downwards_knee_parabolas
            .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let downwards_knee_width_db = params.compressors.downwards.knee_width_db.value();
            for ((ratio, threshold_db), (knee_parabola_scale, knee_parambola_intercept)) in self
                .downwards_ratios
                .iter()
                .zip(self.downwards_thresholds_db.iter())
                .zip(
                    self.downwards_knee_parabola_scale
                        .iter_mut()
                        .zip(self.downwards_knee_parabola_intercept.iter_mut()),
                )
            {
                // This is the formula from the Digital Dynamic Range Compressor Design paper by
                // Dimitrios Giannoulis et. al. These are `a` and `b` from the `x + a * (x + b)^2`
                // respectively used to compute the soft knee respectively.
                (*knee_parabola_scale, *knee_parambola_intercept) =
                    downwards_soft_knee_coefficients(
                        *threshold_db,
                        downwards_knee_width_db,
                        *ratio,
                    );
            }
        }

        if self
            .should_update_upwards_knee_parabolas
            .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let upwards_knee_width_db = params.compressors.upwards.knee_width_db.value();
            for ((ratio, threshold_db), (knee_parabola_scale, knee_parambola_intercept)) in self
                .upwards_ratios
                .iter()
                .zip(self.upwards_thresholds_db.iter())
                .zip(
                    self.upwards_knee_parabola_scale
                        .iter_mut()
                        .zip(self.upwards_knee_parabola_intercept.iter_mut()),
                )
            {
                // The upwards version is slightly different
                (*knee_parabola_scale, *knee_parambola_intercept) =
                    upwards_soft_knee_coefficients(*threshold_db, upwards_knee_width_db, *ratio);
            }
        }
    }
}

/// Apply downwards compression to the input with the supplied parameters. All values are in
/// decibels.
fn compress_downwards(
    input_db: f32,
    threshold_db: f32,
    ratio: f32,
    knee_width_db: f32,
    knee_parabola_scale: f32,
    knee_parabola_intercept: f32,
) -> f32 {
    // The soft-knee option will fade in the compression curve when reaching the knee start until it
    // matches the hard-knee curve at the knee-end
    let knee_start_db = threshold_db - (knee_width_db / 2.0);
    let knee_end_db = threshold_db + (knee_width_db / 2.0);
    if input_db <= knee_start_db {
        input_db
    } else if input_db <= knee_end_db {
        // See the `knee_parabola_intercept` field documentation for the full formula. The entire
        // osft knee part can be skipped if `knee_width_db == 0.0`.
        let parabola_x = input_db + knee_parabola_intercept;
        input_db + (knee_parabola_scale * parabola_x * parabola_x)
    } else {
        threshold_db + ((input_db - threshold_db) / ratio)
    }
}

/// Apply upwards compression to the input with the supplied parameters. All values are in
/// decibels.
fn compress_upwards(
    input_db: f32,
    threshold_db: f32,
    ratio: f32,
    knee_width_db: f32,
    knee_parabola_scale: f32,
    knee_parabola_intercept: f32,
) -> f32 {
    // We'll keep the terminology consistent, start is below the threshold, and end is above the
    // threshold
    let knee_start_db = threshold_db - (knee_width_db / 2.0);
    let knee_end_db = threshold_db + (knee_width_db / 2.0);

    // This goes the other way around compared to the downwards compression
    if input_db >= knee_end_db {
        input_db
    } else if input_db >= knee_start_db {
        let parabola_x = input_db + knee_parabola_intercept;
        input_db + (knee_parabola_scale * parabola_x * parabola_x)
    } else {
        threshold_db + ((input_db - threshold_db) / ratio)
    }
}

/// Compute the `(scale, intercept)`/`(a, b)` coefficients for the parabolic formula `x + a * (x +
/// b)^2`. The formula is taken from the Digital Dynamic Range Compressor Design paper by Dimitrios
/// Giannoulis et. al. This version applies to downwards compression. It can be precalculated for
/// the regular modes, since it's dependent on the threshold it has to be recomputed for every
/// sample with the sidechain matching mode.
fn downwards_soft_knee_coefficients(
    threshold_db: f32,
    knee_width_db: f32,
    ratio: f32,
) -> (f32, f32) {
    let scale = if knee_width_db != 0.0 {
        (2.0 * knee_width_db * ratio).recip() - (2.0 * knee_width_db).recip()
    } else {
        1.0
    };
    let intercept = -threshold_db + (knee_width_db / 2.0);

    (scale, intercept)
}

/// [`downwards_soft_knee_coefficients()`], but for upwards compression.
fn upwards_soft_knee_coefficients(threshold_db: f32, knee_width_db: f32, ratio: f32) -> (f32, f32) {
    // For the upwards version the scale becomes negated
    let scale = if knee_width_db != 0.0 {
        -((2.0 * knee_width_db * ratio).recip() - (2.0 * knee_width_db).recip())
    } else {
        1.0
    };
    // And the `+ (knee/2)` becomes `- (knee/2)` in the intercept
    let intercept = -threshold_db - (knee_width_db / 2.0);

    (scale, intercept)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nih_plug::prelude::{BufferConfig, ProcessMode};
    use triple_buffer::TripleBuffer;

    fn test_buffer_config(window_size: usize) -> BufferConfig {
        BufferConfig {
            sample_rate: 48_000.0,
            min_buffer_size: None,
            max_buffer_size: window_size as u32,
            process_mode: ProcessMode::Realtime,
        }
    }

    fn make_bank_and_params(
        num_channels: usize,
        window_size: usize,
    ) -> (
        CompressorBank,
        SpectralCompressorParams,
        triple_buffer::Output<FrozenIrData>,
    ) {
        let (analyzer_input_data, _analyzer_output_data) =
            TripleBuffer::<AnalyzerData>::default().split();
        let (frozen_ir_input_data, frozen_ir_output_data) =
            TripleBuffer::<FrozenIrData>::default().split();
        let match_curve_runtime = Arc::new(MatchCurveRuntime::new());
        let mut compressor_bank = CompressorBank::new(
            analyzer_input_data,
            frozen_ir_input_data,
            match_curve_runtime,
            num_channels,
            window_size,
        );
        let buffer_config = test_buffer_config(window_size);
        compressor_bank.update_capacity(num_channels, window_size);
        compressor_bank.resize(&buffer_config, window_size);
        configure_test_downwards_compression(&mut compressor_bank, -24.0, 4.0, 0.0);

        let params = SpectralCompressorParams::new(&compressor_bank);

        (compressor_bank, params, frozen_ir_output_data)
    }

    fn configure_test_downwards_compression(
        compressor_bank: &mut CompressorBank,
        threshold_db: f32,
        ratio: f32,
        knee_width_db: f32,
    ) {
        let (knee_scale, knee_intercept) =
            downwards_soft_knee_coefficients(threshold_db, knee_width_db, ratio);

        compressor_bank.downwards_thresholds_db.fill(threshold_db);
        compressor_bank.downwards_ratios.fill(ratio);
        compressor_bank
            .downwards_knee_parabola_scale
            .fill(knee_scale);
        compressor_bank
            .downwards_knee_parabola_intercept
            .fill(knee_intercept);
        compressor_bank.upwards_thresholds_db.fill(0.0);
        compressor_bank.upwards_ratios.fill(1.0);
        compressor_bank.upwards_knee_parabola_scale.fill(0.0);
        compressor_bank.upwards_knee_parabola_intercept.fill(0.0);
    }

    fn make_complex_buffer(scale: f32, len: usize) -> Vec<Complex32> {
        (0..len)
            .map(|bin_idx| Complex32::new(scale * (bin_idx as f32 + 1.0), 0.0))
            .collect()
    }

    fn assert_snapshot_differs(lhs: &[f32], rhs: &[f32]) {
        assert!(
            lhs.iter()
                .zip(rhs.iter())
                .any(|(left, right)| (left - right).abs() > 1.0e-4),
            "expected snapshots to differ"
        );
    }

    fn assert_slice_near(lhs: &[f32], rhs: &[f32]) {
        assert_eq!(lhs.len(), rhs.len());
        for (idx, (left, right)) in lhs.iter().zip(rhs).enumerate() {
            assert!(
                (*left - *right).abs() < 1.0e-5,
                "mismatch at {idx}: {left} != {right}"
            );
        }
    }

    fn run_with_large_stack(test_fn: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(test_fn)
            .expect("failed to spawn test thread")
            .join()
            .expect("test thread panicked");
    }

    #[test]
    fn threshold_slope_default_displays_as_pink_noise_tilt() {
        assert_eq!(threshold_slope_to_display_value(0.0), 3.0);
        assert_eq!(threshold_slope_from_display_value(3.0), 0.0);
        assert_eq!(internal_threshold_slope(0.0), -3.0);
    }

    #[test]
    fn threshold_slope_zero_display_maps_to_white_noise_tilt() {
        let stored_slope = threshold_slope_from_display_value(0.0);

        assert_eq!(stored_slope, 3.0);
        assert_eq!(internal_threshold_slope(stored_slope), 0.0);
    }

    #[test]
    fn gain_smoothing_zero_percent_is_unchanged() {
        let window_size = 32;
        let num_bins = window_size / 2 + 1;
        let (mut compressor_bank, _params, _frozen_ir_output_data) =
            make_bank_and_params(1, window_size);
        let raw: Vec<f32> = (0..num_bins)
            .map(|bin_idx| (bin_idx as f32 % 5.0) - 8.0)
            .collect();
        compressor_bank.raw_gain_difference_db[..num_bins].copy_from_slice(&raw);

        compressor_bank.smooth_gain_differences(0.0, 1, num_bins);

        assert_slice_near(
            &compressor_bank.smoothed_gain_difference_db[..num_bins],
            &raw,
        );
    }

    #[test]
    fn gain_smoothing_reduces_isolated_bin_spikes() {
        let window_size = 64;
        let num_bins = window_size / 2 + 1;
        let (mut compressor_bank, _params, _frozen_ir_output_data) =
            make_bank_and_params(1, window_size);
        compressor_bank.raw_gain_difference_db[..num_bins].fill(0.0);
        compressor_bank.raw_gain_difference_db[16] = -24.0;

        compressor_bank.smooth_gain_differences(1.0, 1, num_bins);

        assert!(compressor_bank.smoothed_gain_difference_db[16] > -24.0);
        assert!(compressor_bank.smoothed_gain_difference_db[16] < 0.0);
        assert!(compressor_bank.smoothed_gain_difference_db[15] < 0.0);
        assert!(compressor_bank.smoothed_gain_difference_db[17] < 0.0);
    }

    #[test]
    fn gain_smoothing_preserves_constant_curves() {
        let window_size = 64;
        let num_bins = window_size / 2 + 1;
        let (mut compressor_bank, _params, _frozen_ir_output_data) =
            make_bank_and_params(1, window_size);
        compressor_bank.raw_gain_difference_db[..num_bins].fill(-6.0);

        compressor_bank.smooth_gain_differences(1.0, 1, num_bins);

        for gain_db in &compressor_bank.smoothed_gain_difference_db[..num_bins] {
            assert!((*gain_db + 6.0).abs() < 1.0e-5);
        }
    }

    #[test]
    fn gain_smoothing_does_not_add_low_frequency_upwards_gain() {
        let window_size = 64;
        let num_bins = window_size / 2 + 1;
        let (mut compressor_bank, _params, _frozen_ir_output_data) =
            make_bank_and_params(1, window_size);
        compressor_bank.raw_gain_difference_db[..num_bins].fill(0.0);
        compressor_bank.raw_gain_difference_db[3] = 18.0;
        compressor_bank.raw_gain_difference_db[4] = 18.0;

        compressor_bank.smooth_gain_differences(1.0, 4, num_bins);

        assert!(compressor_bank.smoothed_gain_difference_db[1] <= 0.0);
        assert!(compressor_bank.smoothed_gain_difference_db[2] <= 0.0);
        assert!(compressor_bank.smoothed_gain_difference_db[3] <= 0.0);
        assert!(compressor_bank.smoothed_gain_difference_db[4] > 0.0);
    }

    #[test]
    fn freeze_captures_smoothed_gain_curve() {
        let window_size = 64;
        let num_bins = window_size / 2 + 1;
        let (mut compressor_bank, _params, _frozen_ir_output_data) =
            make_bank_and_params(1, window_size);
        compressor_bank.raw_gain_difference_db[..num_bins].fill(0.0);
        compressor_bank.raw_gain_difference_db[16] = -24.0;
        compressor_bank.smooth_gain_differences(1.0, 1, num_bins);
        let smoothed_snapshot = compressor_bank.smoothed_gain_difference_db[..num_bins].to_vec();
        let mut buffer = make_complex_buffer(1.0, num_bins);

        compressor_bank.apply_gain_differences(&mut buffer, 0, true, true);

        assert!(compressor_bank.frozen_gain_snapshot_valid[0]);
        assert_slice_near(
            &compressor_bank.frozen_gain_difference_db[0][..num_bins],
            &smoothed_snapshot,
        );
        assert_ne!(
            compressor_bank.frozen_gain_difference_db[0][16],
            compressor_bank.raw_gain_difference_db[16]
        );
    }

    #[test]
    fn freeze_captures_snapshot_on_first_enabled_frame() {
        run_with_large_stack(|| {
            let window_size = 16;
            let num_bins = window_size / 2 + 1;
            let (mut compressor_bank, params, _frozen_ir_output_data) =
                make_bank_and_params(1, window_size);

            let mut warmup = make_complex_buffer(1.0, num_bins);
            compressor_bank.update_envelopes(&warmup, 0, &params, 4);
            compressor_bank.compress(&mut warmup, 0, &params, 1, false, true);
            assert!(!compressor_bank.frozen_gain_snapshot_valid[0]);

            compressor_bank.sync_freeze_state(true);
            let mut capture = make_complex_buffer(2.0, num_bins);
            compressor_bank.update_envelopes(&capture, 0, &params, 4);
            compressor_bank.compress(&mut capture, 0, &params, 1, true, true);

            assert!(compressor_bank.frozen_gain_snapshot_valid[0]);
            assert!(compressor_bank.frozen_gain_difference_db[0]
                .iter()
                .skip(1)
                .any(|gain| gain.abs() > 1.0e-4));
        });
    }

    #[test]
    fn freeze_holds_snapshot_across_input_and_sidechain_changes() {
        run_with_large_stack(|| {
            let window_size = 16;
            let num_bins = window_size / 2 + 1;
            let (mut compressor_bank, params, _frozen_ir_output_data) =
                make_bank_and_params(1, window_size);

            let sidechain_a = make_complex_buffer(1.0, num_bins);
            compressor_bank.process_sidechain(&sidechain_a, 0);
            compressor_bank.sync_freeze_state(true);
            let mut capture = make_complex_buffer(3.0, num_bins);
            compressor_bank.update_envelopes(&capture, 0, &params, 4);
            compressor_bank.compress_sidechain_match(&mut capture, 0, &params, 1, true, true);
            let frozen_snapshot = compressor_bank.frozen_gain_difference_db[0].clone();

            let sidechain_b = make_complex_buffer(0.1, num_bins);
            compressor_bank.process_sidechain(&sidechain_b, 0);
            let input_before = make_complex_buffer(0.25, num_bins);
            let mut held = input_before.clone();
            compressor_bank.update_envelopes(&held, 0, &params, 4);
            compressor_bank.compress_sidechain_match(&mut held, 0, &params, 1, true, true);

            assert_eq!(
                compressor_bank.frozen_gain_difference_db[0],
                frozen_snapshot
            );
            let expected_gain = util::db_to_gain_fast(frozen_snapshot[1]);
            let actual_gain = held[1].norm() / input_before[1].norm();
            assert!((actual_gain - expected_gain).abs() < 1.0e-4);
        });
    }

    #[test]
    fn freeze_recaptures_after_toggle_off_and_on() {
        run_with_large_stack(|| {
            let window_size = 16;
            let num_bins = window_size / 2 + 1;
            let (mut compressor_bank, params, mut frozen_ir_output_data) =
                make_bank_and_params(1, window_size);

            compressor_bank.sync_freeze_state(true);
            let mut first_capture = make_complex_buffer(2.0, num_bins);
            compressor_bank.update_envelopes(&first_capture, 0, &params, 4);
            compressor_bank.compress(&mut first_capture, 0, &params, 1, true, true);
            let snapshot_a = compressor_bank.frozen_gain_difference_db[0].clone();
            compressor_bank.publish_frozen_ir_snapshot(4);
            assert!(frozen_ir_output_data.read().valid);

            compressor_bank.sync_freeze_state(false);
            let mut live = make_complex_buffer(0.3, num_bins);
            compressor_bank.update_envelopes(&live, 0, &params, 4);
            compressor_bank.compress(&mut live, 0, &params, 1, false, true);
            assert!(!compressor_bank.frozen_gain_snapshot_valid[0]);
            assert!(!frozen_ir_output_data.read().valid);

            compressor_bank.sync_freeze_state(true);
            let mut second_capture = make_complex_buffer(6.0, num_bins);
            compressor_bank.update_envelopes(&second_capture, 0, &params, 4);
            compressor_bank.compress(&mut second_capture, 0, &params, 1, true, true);
            let snapshot_b = compressor_bank.frozen_gain_difference_db[0].clone();
            compressor_bank.publish_frozen_ir_snapshot(4);

            assert!(compressor_bank.frozen_gain_snapshot_valid[0]);
            assert_snapshot_differs(&snapshot_a, &snapshot_b);
            assert!(frozen_ir_output_data.read().valid);
        });
    }

    #[test]
    fn freeze_snapshot_is_cleared_on_reset_and_resize() {
        run_with_large_stack(|| {
            let window_size = 16;
            let num_bins = window_size / 2 + 1;
            let (mut compressor_bank, params, mut frozen_ir_output_data) =
                make_bank_and_params(1, window_size);

            compressor_bank.sync_freeze_state(true);
            let mut capture = make_complex_buffer(2.5, num_bins);
            compressor_bank.update_envelopes(&capture, 0, &params, 4);
            compressor_bank.compress(&mut capture, 0, &params, 1, true, true);
            assert!(compressor_bank.frozen_gain_snapshot_valid[0]);
            compressor_bank.publish_frozen_ir_snapshot(4);
            assert!(frozen_ir_output_data.read().valid);

            compressor_bank.reset();
            assert!(!compressor_bank.frozen_gain_snapshot_valid[0]);
            assert!(compressor_bank.frozen_gain_difference_db[0]
                .iter()
                .all(|gain| *gain == 0.0));
            assert!(!frozen_ir_output_data.read().valid);

            compressor_bank.sync_freeze_state(true);
            let mut recapture = make_complex_buffer(4.0, num_bins);
            compressor_bank.update_envelopes(&recapture, 0, &params, 4);
            compressor_bank.compress(&mut recapture, 0, &params, 1, true, true);
            assert!(compressor_bank.frozen_gain_snapshot_valid[0]);
            compressor_bank.publish_frozen_ir_snapshot(4);
            assert!(frozen_ir_output_data.read().valid);

            compressor_bank.resize(&test_buffer_config(32), 32);
            assert!(!compressor_bank.frozen_gain_snapshot_valid[0]);
            assert!(compressor_bank.frozen_gain_difference_db[0]
                .iter()
                .all(|gain| *gain == 0.0));
            let snapshot = frozen_ir_output_data.read().clone();
            assert!(!snapshot.valid);
            assert_eq!(snapshot.window_size, 32);
        });
    }

    #[test]
    fn freeze_publishes_export_snapshot_after_capture() {
        run_with_large_stack(|| {
            let window_size = 16;
            let num_bins = window_size / 2 + 1;
            let (mut compressor_bank, params, mut frozen_ir_output_data) =
                make_bank_and_params(1, window_size);

            compressor_bank.sync_freeze_state(true);
            let mut capture = make_complex_buffer(3.5, num_bins);
            compressor_bank.update_envelopes(&capture, 0, &params, 4);
            compressor_bank.compress(&mut capture, 0, &params, 1, true, true);
            compressor_bank.publish_frozen_ir_snapshot(4);

            let snapshot = frozen_ir_output_data.read().clone();
            assert!(snapshot.valid);
            assert_eq!(snapshot.window_size, window_size);
            assert_eq!(snapshot.overlap_times, 4);
            assert_eq!(snapshot.num_channels, 1);
            assert!(snapshot.gain_difference_db[0][1].abs() > 1.0e-4);
        });
    }

    #[test]
    fn freeze_export_snapshot_clears_on_unsupported_mode() {
        run_with_large_stack(|| {
            let window_size = 16;
            let num_bins = window_size / 2 + 1;
            let (mut compressor_bank, params, mut frozen_ir_output_data) =
                make_bank_and_params(1, window_size);

            compressor_bank.sync_freeze_state(true);
            let mut capture = make_complex_buffer(3.0, num_bins);
            compressor_bank.update_envelopes(&capture, 0, &params, 4);
            compressor_bank.compress(&mut capture, 0, &params, 1, true, true);
            compressor_bank.publish_frozen_ir_snapshot(4);
            assert!(frozen_ir_output_data.read().valid);

            compressor_bank.sync_freeze_state(false);

            let snapshot = frozen_ir_output_data.read().clone();
            assert!(!snapshot.valid);
        });
    }
}
