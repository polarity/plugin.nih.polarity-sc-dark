//! Converts a captured per-bin gain snapshot into a time-domain impulse response and
//! writes it as a floating-point WAV file.

use std::path::Path;

use hound::{SampleFormat, WavSpec, WavWriter};
use nih_plug::prelude::util;
use nih_plug::util::{StftInput, StftInputMut};
use realfft::num_complex::Complex32;
use realfft::RealFftPlanner;

const MAX_IR_CHANNELS: usize = 2;
/// Exported IR length is twice the FFT window the snapshot was captured at.
const IR_LEN_MULT: usize = 2;

#[derive(Debug, Clone)]
/// Snapshot of frozen compressor state used for offline IR rendering.
pub struct FrozenIrData {
    /// Whether the snapshot contains a captured frozen curve that can be rendered.
    pub valid: bool,
    pub sample_rate: f32,
    pub window_size: usize,
    pub overlap_times: usize,
    pub num_channels: usize,
    /// Per-channel, per-bin gain deltas in dB that produce the captured spectral curve.
    ///
    /// Outer length is always `MAX_IR_CHANNELS`; each inner `Vec` is sized to
    /// `MAX_WINDOW_SIZE / 2 + 1`. Heap-allocated to keep `Default::default()` from blowing the
    /// stack (this struct is triple-buffered and constructed during plugin instantiation).
    pub gain_difference_db: [Vec<f32>; MAX_IR_CHANNELS],
}

impl Default for FrozenIrData {
    fn default() -> Self {
        Self {
            valid: false,
            sample_rate: 1.0,
            window_size: 0,
            overlap_times: 0,
            num_channels: 0,
            gain_difference_db: std::array::from_fn(|_| {
                vec![0.0; crate::MAX_WINDOW_SIZE / 2 + 1]
            }),
        }
    }
}

impl FrozenIrData {
    #[inline]
    pub fn num_bins(&self) -> usize {
        self.window_size / 2 + 1
    }

    #[inline]
    pub fn exported_len(&self) -> usize {
        self.window_size * IR_LEN_MULT
    }
}

/// Renders the frozen snapshot into per-channel IR sample buffers. Frozen state is
/// stored as spectral gain deltas, so this runs an offline STFT->gain->iSTFT pass to
/// turn it back into time-domain audio.
pub fn render_frozen_ir(snapshot: &FrozenIrData) -> Result<Vec<Vec<f32>>, String> {
    validate_snapshot(snapshot)?;

    let impulse_offset = snapshot.window_size;
    let render_input_len = snapshot.window_size * (IR_LEN_MULT + 2);
    let exported_len = snapshot.exported_len();

    let mut channels = vec![vec![0.0; render_input_len]; snapshot.num_channels];
    for channel in &mut channels {
        channel[impulse_offset] = 1.0;
    }

    let rendered = process_signal_with_snapshot(snapshot, channels)?;

    let export_start = impulse_offset + snapshot.window_size;
    Ok(rendered
        .into_iter()
        .map(|channel| channel[export_start..export_start + exported_len].to_vec())
        .collect())
}

/// Renders and writes a frozen snapshot as a 32-bit float WAV file at `path`.
pub fn write_frozen_ir_wav(snapshot: &FrozenIrData, path: &Path) -> Result<(), String> {
    let rendered = render_frozen_ir(snapshot)?;
    debug_assert_eq!(rendered.len(), snapshot.num_channels);
    let spec = WavSpec {
        channels: snapshot.num_channels as u16,
        sample_rate: snapshot.sample_rate.round().max(1.0) as u32,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut writer = WavWriter::create(path, spec)
        .map_err(|err| format!("{}: {err}", path.display()))?;
    for sample_idx in 0..rendered[0].len() {
        for channel in rendered.iter().take(snapshot.num_channels) {
            writer
                .write_sample(channel[sample_idx])
                .map_err(|err| format!("{}: {err}", path.display()))?;
        }
    }

    writer
        .finalize()
        .map_err(|err| format!("{}: {err}", path.display()))
}

fn validate_snapshot(snapshot: &FrozenIrData) -> Result<(), String> {
    if !snapshot.valid {
        return Err("snapshot not ready".into());
    }
    if snapshot.num_channels == 0 || snapshot.num_channels > MAX_IR_CHANNELS {
        return Err(format!("bad channel count: {}", snapshot.num_channels));
    }
    if snapshot.window_size == 0 || snapshot.window_size > crate::MAX_WINDOW_SIZE {
        return Err(format!("bad window size: {}", snapshot.window_size));
    }
    if snapshot.overlap_times == 0 {
        return Err("overlap must be non-zero".into());
    }
    if snapshot.sample_rate <= 0.0 {
        return Err(format!("bad sample rate: {}", snapshot.sample_rate));
    }

    Ok(())
}

/// First non-DC bin to keep once we suppress DC and the first few ultra-low bins.
fn first_non_dc_bin_idx(snapshot: &FrozenIrData) -> usize {
    (20.0 / ((snapshot.sample_rate / 2.0) / snapshot.num_bins() as f32)).floor() as usize + 1
}

/// Runs the offline STFT->gain->iSTFT pipeline against a snapshot's bin gains. The
/// input `channels` must match `snapshot.num_channels` and all share the same length.
fn process_signal_with_snapshot(
    snapshot: &FrozenIrData,
    channels: Vec<Vec<f32>>,
) -> Result<Vec<Vec<f32>>, String> {
    validate_snapshot(snapshot)?;

    if channels.len() != snapshot.num_channels {
        return Err(format!(
            "channel count mismatch: {} vs {}",
            snapshot.num_channels,
            channels.len()
        ));
    }

    let Some(buffer_len) = channels.first().map(Vec::len) else {
        return Err("need at least one channel".into());
    };
    if channels.iter().any(|channel| channel.len() != buffer_len) {
        return Err("channels must have equal length".into());
    }

    let num_bins = snapshot.num_bins();
    let first_non_dc_bin_idx = first_non_dc_bin_idx(snapshot);
    let gain_compensation =
        ((snapshot.overlap_times as f32 / 4.0) * 1.5).recip() / snapshot.window_size as f32;
    let input_gain = gain_compensation.sqrt();
    let output_gain = gain_compensation.sqrt();

    let mut planner = RealFftPlanner::new();
    let r2c_plan = planner.plan_fft_forward(snapshot.window_size);
    let c2r_plan = planner.plan_fft_inverse(snapshot.window_size);
    let mut complex_fft_buffer = vec![Complex32::default(); num_bins];
    let mut window_function = vec![0.0; snapshot.window_size];
    util::window::hann_in_place(&mut window_function);

    let mut stft: util::StftHelper<0> =
        util::StftHelper::new(snapshot.num_channels, snapshot.window_size, 0);
    stft.set_block_size(snapshot.window_size);

    let mut rendered_buffer = OfflineStftBuffer { channels };
    let mut processing_error: Option<String> = None;
    stft.process_overlap_add(
        &mut rendered_buffer,
        snapshot.overlap_times,
        |channel_idx, real_fft_buffer| {
            if processing_error.is_some() {
                return;
            }

            for (sample, window_sample) in real_fft_buffer.iter_mut().zip(&window_function) {
                *sample *= window_sample * input_gain;
            }

            if let Err(err) =
                r2c_plan.process_with_scratch(real_fft_buffer, &mut complex_fft_buffer, &mut [])
            {
                processing_error = Some(format!("forward FFT failed: {err}"));
                return;
            }

            for bin in complex_fft_buffer
                .iter_mut()
                .take(first_non_dc_bin_idx.min(num_bins))
            {
                *bin = Complex32::default();
            }

            for (bin, gain_db) in complex_fft_buffer
                .iter_mut()
                .zip(snapshot.gain_difference_db[channel_idx].iter())
            {
                *bin *= util::db_to_gain_fast(*gain_db);
            }

            if let Err(err) =
                c2r_plan.process_with_scratch(&mut complex_fft_buffer, real_fft_buffer, &mut [])
            {
                processing_error = Some(format!("inverse FFT failed: {err}"));
                return;
            }

            for (sample, window_sample) in real_fft_buffer.iter_mut().zip(&window_function) {
                *sample *= window_sample * output_gain;
            }
        },
    );

    if let Some(error) = processing_error {
        return Err(error);
    }

    Ok(rendered_buffer.channels)
}

/// Minimal buffer adapter so `StftHelper` can process owned offline channel data.
struct OfflineStftBuffer {
    channels: Vec<Vec<f32>>,
}

impl StftInput for OfflineStftBuffer {
    fn num_samples(&self) -> usize {
        self.channels.first().map_or(0, Vec::len)
    }

    fn num_channels(&self) -> usize {
        self.channels.len()
    }

    unsafe fn get_sample_unchecked(&self, channel: usize, sample_idx: usize) -> f32 {
        *self
            .channels
            .get_unchecked(channel)
            .get_unchecked(sample_idx)
    }
}

impl StftInputMut for OfflineStftBuffer {
    unsafe fn get_sample_unchecked_mut(&mut self, channel: usize, sample_idx: usize) -> &mut f32 {
        self.channels
            .get_unchecked_mut(channel)
            .get_unchecked_mut(sample_idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn make_snapshot(
        window_size: usize,
        overlap_times: usize,
        num_channels: usize,
    ) -> FrozenIrData {
        let num_bins = window_size / 2 + 1;
        let mut snapshot = FrozenIrData {
            valid: true,
            sample_rate: 48_000.0,
            window_size,
            overlap_times,
            num_channels,
            ..FrozenIrData::default()
        };

        for channel_idx in 0..num_channels {
            snapshot.gain_difference_db[channel_idx][..num_bins].fill(0.0);
        }

        snapshot
    }

    fn transfer_function_db(ir: &[f32], fft_size: usize, bin_idx: usize) -> f32 {
        let mut accumulator = Complex32::default();
        for (sample_idx, sample) in ir.iter().copied().enumerate() {
            let phase = -TAU * bin_idx as f32 * sample_idx as f32 / fft_size as f32;
            let basis = Complex32::new(phase.cos(), phase.sin());
            accumulator += basis * sample;
        }

        util::gain_to_db(accumulator.norm().max(f32::EPSILON))
    }

    fn stable_bin_range(snapshot: &FrozenIrData) -> std::ops::Range<usize> {
        let start = first_non_dc_bin_idx(snapshot)
            .max(64)
            .min(snapshot.num_bins());
        let end = snapshot.num_bins().saturating_sub(8).max(start);
        start..end
    }

    #[test]
    fn unity_renders_to_near_delta() {
        let snapshot = make_snapshot(512, 4, 1);
        let rendered = render_frozen_ir(&snapshot).unwrap();
        let ir = &rendered[0];

        assert_eq!(ir.len(), snapshot.exported_len());
        for bin_idx in stable_bin_range(&snapshot) {
            let actual_db = transfer_function_db(ir, snapshot.window_size, bin_idx);
            assert!(actual_db.abs() < 1.0);
        }
    }

    #[test]
    fn shaped_matches_target_bin_gains() {
        let mut snapshot = make_snapshot(512, 4, 1);
        for bin_idx in 1..snapshot.num_bins() {
            let position = bin_idx as f32 / (snapshot.num_bins() - 1) as f32;
            snapshot.gain_difference_db[0][bin_idx] = -7.5 + position * 10.0;
        }

        let rendered = render_frozen_ir(&snapshot).unwrap();
        let ir = &rendered[0];
        for bin_idx in stable_bin_range(&snapshot) {
            let actual_db = transfer_function_db(ir, snapshot.window_size, bin_idx);
            let expected_db = snapshot.gain_difference_db[0][bin_idx];
            assert!((actual_db - expected_db).abs() < 2.0);
        }
    }

    #[test]
    fn stereo_keeps_per_channel_curves() {
        let mut snapshot = make_snapshot(512, 4, 2);
        for bin_idx in 1..snapshot.num_bins() {
            snapshot.gain_difference_db[0][bin_idx] = -6.0;
            snapshot.gain_difference_db[1][bin_idx] = 3.0;
        }

        let rendered = render_frozen_ir(&snapshot).unwrap();
        assert_eq!(rendered.len(), 2);
        assert_ne!(rendered[0], rendered[1]);

        let probe_bin = stable_bin_range(&snapshot).start + 12;
        let left_db = transfer_function_db(&rendered[0], snapshot.window_size, probe_bin);
        let right_db = transfer_function_db(&rendered[1], snapshot.window_size, probe_bin);
        assert!((left_db + 6.0).abs() < 1.0);
        assert!((right_db - 3.0).abs() < 1.0);
    }
}
