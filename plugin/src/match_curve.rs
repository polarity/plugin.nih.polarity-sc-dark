use atomic_float::AtomicF32;
use nih_plug::prelude::util;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use crate::curve::{
    CurvePoint, MAX_THRESHOLD_CURVE_POINTS, THRESHOLD_CURVE_MAX_FREQUENCY_HZ,
    THRESHOLD_CURVE_MIN_FREQUENCY_HZ, THRESHOLD_CURVE_POINT_OFFSET_LIMIT_DB,
};

pub(crate) const MATCH_CURVE_DURATION_SECONDS: f32 = 5.0;

const SILENCE_FLOOR_DB: f32 = -120.0;
const MIN_VALID_FREQUENCY_HZ: f32 = THRESHOLD_CURVE_MIN_FREQUENCY_HZ;
const MAX_FIT_RESIDUAL_DB: f32 = 1.5;
const MAX_NODE_RESIDUAL_DB: f32 = 3.0;
const MIN_NODE_DISTANCE_LN: f32 = 0.22;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum MatchCurveState {
    Idle = 0,
    Requested = 1,
    Running = 2,
    Ready = 3,
    Failed = 4,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MatchCurveFit {
    pub(crate) intercept: f32,
    pub(crate) center_frequency: f32,
    pub(crate) slope: f32,
    pub(crate) curve: f32,
    pub(crate) points: [CurvePoint; MAX_THRESHOLD_CURVE_POINTS],
}

impl Default for MatchCurveFit {
    fn default() -> Self {
        Self {
            intercept: 0.0,
            center_frequency: 1_000.0,
            slope: 0.0,
            curve: 0.0,
            points: [CurvePoint::default(); MAX_THRESHOLD_CURVE_POINTS],
        }
    }
}

#[derive(Debug)]
pub(crate) enum MatchCurveResult {
    Matched(MatchCurveFit),
    Failed,
}

pub(crate) struct MatchCurveRuntime {
    state: AtomicU8,
    intercept: AtomicF32,
    center_frequency: AtomicF32,
    slope: AtomicF32,
    curve: AtomicF32,
    point_enabled: [AtomicBool; MAX_THRESHOLD_CURVE_POINTS],
    point_frequency: [AtomicF32; MAX_THRESHOLD_CURVE_POINTS],
    point_offset_db: [AtomicF32; MAX_THRESHOLD_CURVE_POINTS],
}

impl MatchCurveRuntime {
    pub(crate) fn new() -> Self {
        Self {
            state: AtomicU8::new(MatchCurveState::Idle as u8),
            intercept: AtomicF32::new(0.0),
            center_frequency: AtomicF32::new(1_000.0),
            slope: AtomicF32::new(0.0),
            curve: AtomicF32::new(0.0),
            point_enabled: std::array::from_fn(|_| AtomicBool::new(false)),
            point_frequency: std::array::from_fn(|_| AtomicF32::new(1_000.0)),
            point_offset_db: std::array::from_fn(|_| AtomicF32::new(0.0)),
        }
    }

    pub(crate) fn request(&self) {
        self.state
            .store(MatchCurveState::Requested as u8, Ordering::Release);
    }

    pub(crate) fn begin_running_if_requested(&self) -> bool {
        self.state
            .compare_exchange(
                MatchCurveState::Requested as u8,
                MatchCurveState::Running as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    pub(crate) fn publish_result(&self, result: Option<MatchCurveFit>) {
        match result {
            Some(fit) => {
                self.intercept.store(fit.intercept, Ordering::Release);
                self.center_frequency
                    .store(fit.center_frequency, Ordering::Release);
                self.slope.store(fit.slope, Ordering::Release);
                self.curve.store(fit.curve, Ordering::Release);
                for (index, point) in fit.points.iter().enumerate() {
                    self.point_frequency[index].store(point.frequency, Ordering::Release);
                    self.point_offset_db[index].store(point.offset_db, Ordering::Release);
                    self.point_enabled[index].store(point.enabled, Ordering::Release);
                }
                self.state
                    .store(MatchCurveState::Ready as u8, Ordering::Release);
            }
            None => {
                self.state
                    .store(MatchCurveState::Failed as u8, Ordering::Release);
            }
        }
    }

    pub(crate) fn take_finished_result(&self) -> Option<MatchCurveResult> {
        let state = self.state.load(Ordering::Acquire);
        match state {
            value if value == MatchCurveState::Ready as u8 => {
                if self
                    .state
                    .compare_exchange(
                        MatchCurveState::Ready as u8,
                        MatchCurveState::Idle as u8,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
                {
                    Some(MatchCurveResult::Matched(self.load_fit()))
                } else {
                    None
                }
            }
            value if value == MatchCurveState::Failed as u8 => self
                .state
                .compare_exchange(
                    MatchCurveState::Failed as u8,
                    MatchCurveState::Idle as u8,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .ok()
                .map(|_| MatchCurveResult::Failed),
            _ => None,
        }
    }

    fn load_fit(&self) -> MatchCurveFit {
        MatchCurveFit {
            intercept: self.intercept.load(Ordering::Acquire),
            center_frequency: self.center_frequency.load(Ordering::Acquire),
            slope: self.slope.load(Ordering::Acquire),
            curve: self.curve.load(Ordering::Acquire),
            points: std::array::from_fn(|index| CurvePoint {
                enabled: self.point_enabled[index].load(Ordering::Acquire),
                frequency: self.point_frequency[index].load(Ordering::Acquire),
                offset_db: self.point_offset_db[index].load(Ordering::Acquire),
            }),
        }
    }
}

pub(crate) struct MatchCurveMeter {
    magnitude_sums: Vec<f64>,
    spectrum_count: u64,
    measured_frames: u64,
    target_frames: u64,
}

impl MatchCurveMeter {
    pub(crate) fn new(max_num_bins: usize) -> Self {
        Self {
            magnitude_sums: vec![0.0; max_num_bins],
            spectrum_count: 0,
            measured_frames: 0,
            target_frames: 0,
        }
    }

    pub(crate) fn resize(&mut self, num_bins: usize) {
        self.magnitude_sums.resize(num_bins, 0.0);
    }

    pub(crate) fn start(&mut self, sample_rate: f32) {
        self.magnitude_sums.fill(0.0);
        self.spectrum_count = 0;
        self.measured_frames = 0;
        self.target_frames = (sample_rate * MATCH_CURVE_DURATION_SECONDS)
            .round()
            .max(1.0) as u64;
    }

    pub(crate) fn is_active(&self) -> bool {
        self.target_frames > 0 && self.measured_frames < self.target_frames
    }

    pub(crate) fn measure_magnitudes<I>(&mut self, magnitudes: I)
    where
        I: IntoIterator<Item = f32>,
    {
        if !self.is_active() {
            return;
        }

        for (sum, magnitude) in self.magnitude_sums.iter_mut().zip(magnitudes) {
            if magnitude.is_finite() && magnitude > 0.0 {
                *sum += magnitude as f64;
            }
        }
        self.spectrum_count += 1;
    }

    pub(crate) fn advance_and_finish(
        &mut self,
        frames: usize,
        ln_freqs: &[f32],
        downwards_offset_db: f32,
        fixed_center_frequency: f32,
        fixed_slope: f32,
    ) -> Option<Option<MatchCurveFit>> {
        if frames == 0 || !self.is_active() {
            return None;
        }

        self.measured_frames = (self.measured_frames + frames as u64).min(self.target_frames);
        if self.measured_frames < self.target_frames {
            return None;
        }

        self.target_frames = 0;
        Some(fit_average_curve_constrained(
            ln_freqs,
            &self.magnitude_sums,
            self.spectrum_count,
            downwards_offset_db,
            fixed_center_frequency,
            fixed_slope,
        ))
    }
}

pub(crate) fn fit_average_curve_constrained(
    ln_freqs: &[f32],
    magnitude_sums: &[f64],
    spectrum_count: u64,
    downwards_offset_db: f32,
    fixed_center_frequency: f32,
    fixed_slope: f32,
) -> Option<MatchCurveFit> {
    let samples = collect_fit_samples(
        ln_freqs,
        magnitude_sums,
        spectrum_count,
        downwards_offset_db,
    )?;
    let center_frequency = fixed_center_frequency.clamp(
        THRESHOLD_CURVE_MIN_FREQUENCY_HZ,
        THRESHOLD_CURVE_MAX_FREQUENCY_HZ,
    );
    let center_ln = center_frequency.ln();
    let (intercept, curve) = fit_quadratic_fixed_center_slope(&samples, center_ln, fixed_slope)?;
    let points = fit_residual_points(&samples, center_ln, intercept, fixed_slope, curve);

    Some(MatchCurveFit {
        intercept,
        center_frequency,
        slope: fixed_slope,
        curve,
        points,
    })
}

#[cfg(test)]
fn fit_average_curve(
    ln_freqs: &[f32],
    magnitude_sums: &[f64],
    spectrum_count: u64,
    downwards_offset_db: f32,
) -> Option<MatchCurveFit> {
    let samples = collect_fit_samples(
        ln_freqs,
        magnitude_sums,
        spectrum_count,
        downwards_offset_db,
    )?;
    let center_ln = mean_ln_frequency(&samples).clamp(
        THRESHOLD_CURVE_MIN_FREQUENCY_HZ.ln(),
        THRESHOLD_CURVE_MAX_FREQUENCY_HZ.ln(),
    );
    let (intercept, slope, curve) = fit_quadratic(&samples, center_ln)?;
    let points = fit_residual_points(&samples, center_ln, intercept, slope, curve);

    Some(MatchCurveFit {
        intercept,
        center_frequency: center_ln.exp(),
        slope,
        curve,
        points,
    })
}

fn collect_fit_samples(
    ln_freqs: &[f32],
    magnitude_sums: &[f64],
    spectrum_count: u64,
    downwards_offset_db: f32,
) -> Option<Vec<FitSample>> {
    if spectrum_count == 0 {
        return None;
    }

    let mut samples = Vec::new();
    let count = spectrum_count as f64;
    for (&ln_freq, &magnitude_sum) in ln_freqs.iter().zip(magnitude_sums).skip(1) {
        let frequency = ln_freq.exp();
        if !ln_freq.is_finite()
            || !(MIN_VALID_FREQUENCY_HZ..=THRESHOLD_CURVE_MAX_FREQUENCY_HZ).contains(&frequency)
        {
            continue;
        }

        let average_magnitude = (magnitude_sum / count) as f32;
        if !average_magnitude.is_finite() || average_magnitude <= 0.0 {
            continue;
        }

        let average_db = util::gain_to_db_fast_epsilon(average_magnitude) - downwards_offset_db;
        if average_db <= SILENCE_FLOOR_DB {
            continue;
        }

        samples.push(FitSample {
            ln_freq,
            target_db: average_db,
        });
    }

    if samples.len() < 3 {
        return None;
    }

    Some(samples)
}

#[derive(Debug, Clone, Copy)]
struct FitSample {
    ln_freq: f32,
    target_db: f32,
}

#[cfg(test)]
fn mean_ln_frequency(samples: &[FitSample]) -> f32 {
    samples.iter().map(|sample| sample.ln_freq).sum::<f32>() / samples.len() as f32
}

#[cfg(test)]
fn fit_quadratic(samples: &[FitSample], center_ln: f32) -> Option<(f32, f32, f32)> {
    let mut sx0 = 0.0f64;
    let mut sx1 = 0.0f64;
    let mut sx2 = 0.0f64;
    let mut sx3 = 0.0f64;
    let mut sx4 = 0.0f64;
    let mut sy0 = 0.0f64;
    let mut sy1 = 0.0f64;
    let mut sy2 = 0.0f64;

    for sample in samples {
        let x = (sample.ln_freq - center_ln) as f64;
        let x2 = x * x;
        let y = sample.target_db as f64;

        sx0 += 1.0;
        sx1 += x;
        sx2 += x2;
        sx3 += x2 * x;
        sx4 += x2 * x2;
        sy0 += y;
        sy1 += y * x;
        sy2 += y * x2;
    }

    solve_3x3(
        [[sx0, sx1, sx2], [sx1, sx2, sx3], [sx2, sx3, sx4]],
        [sy0, sy1, sy2],
    )
    .map(|solution| (solution[0] as f32, solution[1] as f32, solution[2] as f32))
}

fn fit_quadratic_fixed_center_slope(
    samples: &[FitSample],
    center_ln: f32,
    slope: f32,
) -> Option<(f32, f32)> {
    let mut sx0 = 0.0f64;
    let mut sx2 = 0.0f64;
    let mut sx4 = 0.0f64;
    let mut sy0 = 0.0f64;
    let mut sy2 = 0.0f64;

    for sample in samples {
        let x = (sample.ln_freq - center_ln) as f64;
        let x2 = x * x;
        let y = sample.target_db as f64 - (slope as f64 * x);

        sx0 += 1.0;
        sx2 += x2;
        sx4 += x2 * x2;
        sy0 += y;
        sy2 += y * x2;
    }

    let determinant = (sx0 * sx4) - (sx2 * sx2);
    if determinant.abs() <= f64::EPSILON {
        return None;
    }

    let intercept = ((sy0 * sx4) - (sy2 * sx2)) / determinant;
    let curve = ((sx0 * sy2) - (sx2 * sy0)) / determinant;
    Some((intercept as f32, curve as f32))
}

#[cfg(test)]
fn solve_3x3(mut matrix: [[f64; 3]; 3], mut rhs: [f64; 3]) -> Option<[f64; 3]> {
    for pivot_idx in 0..3 {
        let mut best_row = pivot_idx;
        let mut best_value = matrix[pivot_idx][pivot_idx].abs();
        for (row_idx, row) in matrix.iter().enumerate().skip(pivot_idx + 1) {
            let value = row[pivot_idx].abs();
            if value > best_value {
                best_row = row_idx;
                best_value = value;
            }
        }

        if best_value <= f64::EPSILON {
            return None;
        }

        if best_row != pivot_idx {
            matrix.swap(pivot_idx, best_row);
            rhs.swap(pivot_idx, best_row);
        }

        let pivot = matrix[pivot_idx][pivot_idx];
        for col_idx in pivot_idx..3 {
            matrix[pivot_idx][col_idx] /= pivot;
        }
        rhs[pivot_idx] /= pivot;

        for row_idx in 0..3 {
            if row_idx == pivot_idx {
                continue;
            }

            let factor = matrix[row_idx][pivot_idx];
            for col_idx in pivot_idx..3 {
                matrix[row_idx][col_idx] -= factor * matrix[pivot_idx][col_idx];
            }
            rhs[row_idx] -= factor * rhs[pivot_idx];
        }
    }

    Some(rhs)
}

fn fit_residual_points(
    samples: &[FitSample],
    center_ln: f32,
    intercept: f32,
    slope: f32,
    curve: f32,
) -> [CurvePoint; MAX_THRESHOLD_CURVE_POINTS] {
    let mut points = [CurvePoint::default(); MAX_THRESHOLD_CURVE_POINTS];
    let residuals: Vec<(f32, f32)> = samples
        .iter()
        .map(|sample| {
            let x = sample.ln_freq - center_ln;
            (
                sample.ln_freq,
                sample.target_db - (intercept + (slope * x) + (curve * x * x)),
            )
        })
        .collect();

    let max_base_residual = residuals
        .iter()
        .map(|(_, residual)| residual.abs())
        .fold(0.0f32, f32::max);
    if max_base_residual <= MAX_FIT_RESIDUAL_DB {
        return points;
    }

    let mut selected: Vec<(f32, f32)> = Vec::new();
    for point in &mut points {
        let Some((ln_freq, offset_db)) = strongest_residual(&residuals, &selected) else {
            break;
        };

        if offset_db.abs() <= MAX_NODE_RESIDUAL_DB {
            break;
        }

        selected.push((ln_freq, offset_db));
        selected.sort_by(|(left_ln, _), (right_ln, _)| left_ln.total_cmp(right_ln));
        *point = CurvePoint {
            enabled: true,
            frequency: ln_freq.exp().clamp(
                THRESHOLD_CURVE_MIN_FREQUENCY_HZ,
                THRESHOLD_CURVE_MAX_FREQUENCY_HZ,
            ),
            offset_db: offset_db.clamp(
                -THRESHOLD_CURVE_POINT_OFFSET_LIMIT_DB,
                THRESHOLD_CURVE_POINT_OFFSET_LIMIT_DB,
            ),
        };
    }

    points
}

fn strongest_residual(residuals: &[(f32, f32)], selected: &[(f32, f32)]) -> Option<(f32, f32)> {
    residuals
        .iter()
        .copied()
        .filter(|(ln_freq, _)| {
            selected.iter().all(|(selected_ln_freq, _)| {
                (ln_freq - selected_ln_freq).abs() >= MIN_NODE_DISTANCE_LN
            })
        })
        .max_by(|(_, left), (_, right)| left.abs().total_cmp(&right.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ln_freqs_for_bins(num_bins: usize) -> Vec<f32> {
        (0..num_bins)
            .map(|bin_idx| {
                if bin_idx == 0 {
                    0.0
                } else {
                    ((bin_idx as f32 / (num_bins - 1) as f32) * 22_000.0).ln()
                }
            })
            .collect()
    }

    fn sums_from_db(ln_freqs: &[f32], make_db: impl Fn(f32) -> f32) -> Vec<f64> {
        ln_freqs
            .iter()
            .map(|&ln_freq| {
                if ln_freq == 0.0 {
                    0.0
                } else {
                    util::db_to_gain(make_db(ln_freq)) as f64
                }
            })
            .collect()
    }

    #[test]
    fn flat_spectrum_produces_flat_curve_without_nodes() {
        let ln_freqs = ln_freqs_for_bins(256);
        let sums = sums_from_db(&ln_freqs, |_| -18.0);

        let fit = fit_average_curve(&ln_freqs, &sums, 1, 0.0).unwrap();

        assert!((fit.intercept + 18.0).abs() < 1.0e-3);
        assert!(fit.slope.abs() < 1.0e-3);
        assert!(fit.curve.abs() < 1.0e-3);
        assert!(fit.points.iter().all(|point| !point.enabled));
    }

    #[test]
    fn sloped_spectrum_is_represented_by_base_slope() {
        let ln_freqs = ln_freqs_for_bins(256);
        let center = 1_000.0f32.ln();
        let sums = sums_from_db(&ln_freqs, |ln_freq| -24.0 + ((ln_freq - center) * 4.0));

        let fit = fit_average_curve(&ln_freqs, &sums, 1, 0.0).unwrap();

        assert!((fit.slope - 4.0).abs() < 1.0e-3);
        assert!(fit.points.iter().all(|point| !point.enabled));
    }

    #[test]
    fn constrained_fit_preserves_fixed_slope() {
        let ln_freqs = ln_freqs_for_bins(256);
        let center = 1_000.0f32;
        let sums = sums_from_db(&ln_freqs, |ln_freq| -24.0 + ((ln_freq - center.ln()) * 4.0));

        let fit = fit_average_curve_constrained(&ln_freqs, &sums, 1, 0.0, center, 1.5).unwrap();

        assert!((fit.slope - 1.5).abs() < 1.0e-6);
        assert!((fit.center_frequency - center).abs() < 1.0e-3);
    }

    #[test]
    fn constrained_fit_preserves_fixed_center_frequency() {
        let ln_freqs = ln_freqs_for_bins(256);
        let source_center = 2_000.0f32.ln();
        let fixed_center = 750.0f32;
        let sums = sums_from_db(&ln_freqs, |ln_freq| {
            let x = ln_freq - source_center;
            -21.0 + (2.0 * x) + (1.25 * x * x)
        });

        let fit =
            fit_average_curve_constrained(&ln_freqs, &sums, 1, 0.0, fixed_center, 2.0).unwrap();

        assert!((fit.center_frequency - fixed_center).abs() < 1.0e-3);
        assert!((fit.slope - 2.0).abs() < 1.0e-6);
    }

    #[test]
    fn constrained_fit_solves_intercept_and_curve() {
        let ln_freqs = ln_freqs_for_bins(256);
        let center = 1_200.0f32;
        let slope = -2.0f32;
        let curve = 1.75f32;
        let sums = sums_from_db(&ln_freqs, |ln_freq| {
            let x = ln_freq - center.ln();
            -18.0 + (slope * x) + (curve * x * x)
        });

        let fit = fit_average_curve_constrained(&ln_freqs, &sums, 1, 0.0, center, slope).unwrap();

        assert!((fit.intercept + 18.0).abs() < 1.0e-3);
        assert!((fit.curve - curve).abs() < 1.0e-3);
        assert!(fit.points.iter().all(|point| !point.enabled));
    }

    #[test]
    fn residual_peaks_add_limited_nodes() {
        let ln_freqs = ln_freqs_for_bins(512);
        let peak_a = 620.0f32.ln();
        let peak_b = 6_200.0f32.ln();
        let sums = sums_from_db(&ln_freqs, |ln_freq| {
            let a = if (ln_freq - peak_a).abs() < 0.035 {
                12.0
            } else {
                0.0
            };
            let b = if (ln_freq - peak_b).abs() < 0.035 {
                -10.0
            } else {
                0.0
            };
            -30.0 + a + b
        });

        let fit = fit_average_curve(&ln_freqs, &sums, 1, 0.0).unwrap();
        let enabled: Vec<_> = fit.points.iter().filter(|point| point.enabled).collect();

        assert!(!enabled.is_empty());
        assert!(enabled.len() <= MAX_THRESHOLD_CURVE_POINTS);
        assert!(enabled
            .iter()
            .any(|point| (point.frequency.ln() - peak_a).abs() < 0.2));
    }

    #[test]
    fn unused_point_slots_are_disabled() {
        let ln_freqs = ln_freqs_for_bins(128);
        let sums = sums_from_db(&ln_freqs, |ln_freq| {
            -20.0
                + if (ln_freq - 2_000.0f32.ln()).abs() < 0.05 {
                    8.0
                } else {
                    0.0
                }
        });

        let fit = fit_average_curve(&ln_freqs, &sums, 1, 0.0).unwrap();
        let first_disabled = fit
            .points
            .iter()
            .position(|point| !point.enabled)
            .unwrap_or(MAX_THRESHOLD_CURVE_POINTS);

        assert!(fit.points[first_disabled..]
            .iter()
            .all(|point| !point.enabled));
    }

    #[test]
    fn silent_input_returns_failure() {
        let ln_freqs = ln_freqs_for_bins(128);
        let sums = vec![0.0; ln_freqs.len()];

        assert!(fit_average_curve(&ln_freqs, &sums, 16, 0.0).is_none());
    }

    #[test]
    fn accumulator_collects_duration_and_averages_channels() {
        let ln_freqs = ln_freqs_for_bins(64);
        let mut meter = MatchCurveMeter::new(64);
        meter.start(10.0);
        meter.measure_magnitudes([1.0; 64]);
        assert!(meter
            .advance_and_finish(4, &ln_freqs, 0.0, 1_000.0, 0.0)
            .is_none());

        meter.measure_magnitudes([3.0; 64]);
        let fit = meter
            .advance_and_finish(46, &ln_freqs, 0.0, 1_000.0, 0.0)
            .unwrap()
            .unwrap();
        let expected_db = util::gain_to_db(2.0);

        assert!((fit.intercept - expected_db).abs() < 1.0e-3);
    }
}
