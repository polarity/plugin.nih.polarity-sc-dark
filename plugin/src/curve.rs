//! Abstractions for the parameterized threshold curve.
//!
//! This was previously computed directly inside of the `CompressorBank` but this makes it easier to
//! reuse it when drawing the GUI.

pub const MAX_THRESHOLD_CURVE_POINTS: usize = 8;
pub const THRESHOLD_CURVE_MIN_FREQUENCY_HZ: f32 = 30.0;
pub const THRESHOLD_CURVE_MAX_FREQUENCY_HZ: f32 = 22_000.0;
pub const THRESHOLD_CURVE_POINT_OFFSET_LIMIT_DB: f32 = 60.0;

#[derive(Debug, Default, Clone, Copy)]
pub struct CurvePoint {
    pub enabled: bool,
    pub frequency: f32,
    pub offset_db: f32,
}

/// Parameters for a curve, similar to the fields found in `ThresholdParams` but using plain floats
/// instead of parameters.
#[derive(Debug, Default, Clone, Copy)]
pub struct CurveParams {
    /// The compressor threshold at the center frequency. When sidechaining is enabled, the input
    /// signal is gained by the inverse of this value. This replaces the input gain in the original
    /// Spectral Compressor. In the polynomial below, this is the intercept.
    pub intercept: f32,
    /// The center frequency for the target curve when sidechaining is not enabled. The curve is a
    /// polynomial `threshold_db + curve_slope*x + curve_curve*(x^2)` that evaluates to a decibel
    /// value, where `x = ln(center_frequency) - ln(bin_frequency)`. In other words, this is
    /// evaluated in the log/log domain for decibels and octaves.
    pub center_frequency: f32,
    /// The slope for the curve, in the log/log domain. See the polynomial above.
    pub slope: f32,
    /// The, uh, 'curve' for the curve, in the logarithmic domain. This is the third coefficient in
    /// the quadratic polynomial and controls the parabolic behavior. Positive values turn the curve
    /// into a v-shaped curve, while negative values attenuate everything outside of the center
    /// frequency. See the polynomial above.
    pub curve: f32,
    /// User-editable point offsets layered on top of the legacy quadratic curve.
    pub points: [CurvePoint; MAX_THRESHOLD_CURVE_POINTS],
}

/// Evaluates the quadratic threshold curve. This used to be calculated directly inside of the
/// compressor bank since it's so simple, but the editor also needs to compute this so it makes
/// sense to deduplicate it a bit.
///
/// The curve is evaluated in log-log space (so with octaves being the independent variable and gain
/// in decibels being the output of the equation).
pub struct Curve<'a> {
    params: &'a CurveParams,
    /// The natural logarithm of [`CurveParams::center_frequency`].
    ln_center_frequency: f32,
}

impl<'a> Curve<'a> {
    pub fn new(params: &'a CurveParams) -> Self {
        Self {
            params,
            ln_center_frequency: params.center_frequency.ln(),
        }
    }

    /// Evaluate the curve for the natural logarithm of the frequency value. This can be used as an
    /// optimization to avoid computing these logarithms all the time.
    #[inline]
    pub fn evaluate_ln(&self, ln_freq: f32) -> f32 {
        self.evaluate_base_ln(ln_freq) + self.point_offset_ln(ln_freq)
    }

    #[inline]
    pub fn evaluate_base_ln(&self, ln_freq: f32) -> f32 {
        let offset = ln_freq - self.ln_center_frequency;
        self.params.intercept + (self.params.slope * offset) + (self.params.curve * offset * offset)
    }

    #[inline]
    pub fn point_offset_ln(&self, ln_freq: f32) -> f32 {
        let mut left: Option<(f32, f32)> = None;
        let mut right: Option<(f32, f32)> = None;

        for point in self.params.points {
            if !point.enabled || !point.frequency.is_finite() || !point.offset_db.is_finite() {
                continue;
            }

            let point_ln_freq = point
                .frequency
                .clamp(
                    THRESHOLD_CURVE_MIN_FREQUENCY_HZ,
                    THRESHOLD_CURVE_MAX_FREQUENCY_HZ,
                )
                .ln();

            if point_ln_freq <= ln_freq
                && left
                    .map(|(left_ln_freq, _)| point_ln_freq > left_ln_freq)
                    .unwrap_or(true)
            {
                left = Some((point_ln_freq, point.offset_db));
            }

            if point_ln_freq >= ln_freq
                && right
                    .map(|(right_ln_freq, _)| point_ln_freq < right_ln_freq)
                    .unwrap_or(true)
            {
                right = Some((point_ln_freq, point.offset_db));
            }
        }

        match (left, right) {
            (None, None) => 0.0,
            (Some((_, offset_db)), None) | (None, Some((_, offset_db))) => offset_db,
            (Some((left_ln_freq, left_offset_db)), Some((right_ln_freq, right_offset_db))) => {
                if (right_ln_freq - left_ln_freq).abs() <= f32::EPSILON {
                    left_offset_db
                } else {
                    let t =
                        ((ln_freq - left_ln_freq) / (right_ln_freq - left_ln_freq)).clamp(0.0, 1.0);
                    let smooth_t = t * t * (3.0 - (2.0 * t));
                    left_offset_db + ((right_offset_db - left_offset_db) * smooth_t)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params_with_points(points: [CurvePoint; MAX_THRESHOLD_CURVE_POINTS]) -> CurveParams {
        CurveParams {
            intercept: -12.0,
            center_frequency: 1_000.0,
            slope: 0.0,
            curve: 0.0,
            points,
        }
    }

    fn point(frequency: f32, offset_db: f32) -> CurvePoint {
        CurvePoint {
            enabled: true,
            frequency,
            offset_db,
        }
    }

    #[test]
    fn no_points_uses_legacy_curve() {
        let params = CurveParams {
            intercept: -12.0,
            center_frequency: 1_000.0,
            slope: 2.0,
            curve: 1.5,
            points: [CurvePoint::default(); MAX_THRESHOLD_CURVE_POINTS],
        };
        let curve = Curve::new(&params);
        let ln_freq = 2_000.0f32.ln();

        assert_eq!(curve.evaluate_ln(ln_freq), curve.evaluate_base_ln(ln_freq));
    }

    #[test]
    fn one_point_applies_constant_offset() {
        let mut points = [CurvePoint::default(); MAX_THRESHOLD_CURVE_POINTS];
        points[0] = point(1_000.0, 6.0);

        let params = params_with_points(points);
        let curve = Curve::new(&params);

        assert_eq!(curve.point_offset_ln(100.0f32.ln()), 6.0);
        assert_eq!(curve.point_offset_ln(10_000.0f32.ln()), 6.0);
    }

    #[test]
    fn points_do_not_need_to_be_sorted() {
        let mut points = [CurvePoint::default(); MAX_THRESHOLD_CURVE_POINTS];
        points[0] = point(8_000.0, 12.0);
        points[1] = point(500.0, -12.0);

        let params = params_with_points(points);
        let curve = Curve::new(&params);

        assert_eq!(curve.point_offset_ln(500.0f32.ln()), -12.0);
        assert_eq!(curve.point_offset_ln(8_000.0f32.ln()), 12.0);
    }

    #[test]
    fn disabled_points_are_ignored() {
        let mut points = [CurvePoint::default(); MAX_THRESHOLD_CURVE_POINTS];
        points[0] = CurvePoint {
            enabled: false,
            frequency: 1_000.0,
            offset_db: 24.0,
        };

        let params = params_with_points(points);
        let curve = Curve::new(&params);

        assert_eq!(curve.point_offset_ln(1_000.0f32.ln()), 0.0);
    }

    #[test]
    fn point_offsets_clamp_to_edge_points() {
        let mut points = [CurvePoint::default(); MAX_THRESHOLD_CURVE_POINTS];
        points[0] = point(500.0, -9.0);
        points[1] = point(8_000.0, 9.0);

        let params = params_with_points(points);
        let curve = Curve::new(&params);

        assert_eq!(curve.point_offset_ln(100.0f32.ln()), -9.0);
        assert_eq!(curve.point_offset_ln(20_000.0f32.ln()), 9.0);
    }

    #[test]
    fn legacy_curve_still_affects_curve_without_points() {
        let flat_params = CurveParams {
            intercept: -12.0,
            center_frequency: 1_000.0,
            slope: 0.0,
            curve: 0.0,
            points: [CurvePoint::default(); MAX_THRESHOLD_CURVE_POINTS],
        };
        let curved_params = CurveParams {
            curve: 6.0,
            ..flat_params
        };
        let ln_freq = 4_000.0f32.ln();

        assert_ne!(
            Curve::new(&flat_params).evaluate_ln(ln_freq),
            Curve::new(&curved_params).evaluate_ln(ln_freq)
        );
    }
}
