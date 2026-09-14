//! View-only damping of outlier input-token counts for savings analytics.
//!
//! A single runaway command (a `rg` that emits ~26M tokens of matches) otherwise
//! dominates every lifetime figure `rtk gain` and `rtk cc-economics` report: the
//! raw input count is real, but it was never a real token cost, because Claude
//! Code truncates terminal output at ~50KB (~12,500 tokens) before the model ever
//! sees it. Counting all 26M as "saved" turns the savings metric into a record of
//! which command happened to be loudest.
//!
//! Damping compresses everything above a ceiling logarithmically, so outliers
//! still rank above ordinary commands (the function is monotonic — a bigger raw
//! input never damps to a smaller effective one) without swamping them.
//!
//! # Strictly view-only
//!
//! Nothing here touches the SQLite rows. `commands.input_tokens` stays the raw
//! ground truth forever; damping is applied during aggregation, on read. Set
//! `tracking.damping_ceiling = 0` and every reported number reverts exactly to
//! the raw values, because damping is a pure function applied at query time.
//!
//! # The function
//!
//! For ceiling `C`, raw input `I`:
//!
//! ```text
//! I_eff = I                             if C == 0 or I <= C
//! I_eff = round(C * (1 + ln(I / C)))    if I > C
//! ```
//!
//! - **Continuous at the boundary**: at `I == C`, `ln(1) == 0`, so `I_eff == C` —
//!   no cliff where the ceiling kicks in.
//! - **Smooth at the boundary**: the derivative approaches `C/I == 1.0` as
//!   `I -> C+`, matching the identity branch's slope, so the two branches meet
//!   without a kink.
//! - **Monotonic**: `d/dI = C/I > 0` everywhere, so ordering by tokens saved is
//!   preserved. (The rounded integer result is non-decreasing rather than
//!   strictly increasing — above the ceiling the slope is below 1, so adjacent
//!   raw inputs can share an effective value.)
//!
//! At the default ceiling, 26,000,000 raw tokens damp to 108,002 — still by far
//! the largest entry in the table, but ~8.6x the ceiling instead of ~2080x.

/// Default damping ceiling, in tokens.
///
/// Tracks Claude Code's ~50KB terminal-output truncation limit at RTK's
/// `bytes / 4` token estimate (see `tracking::estimate_tokens`): output beyond
/// this never reaches the model, so input above it was never a token cost that
/// RTK could have saved.
pub const DEFAULT_DAMPING_CEILING: usize = 12_500;

/// Applies the view-only damping curve to raw input-token counts.
///
/// Constructed from `tracking.damping_ceiling` (see
/// [`TrackingConfig::to_damper`](crate::core::config::TrackingConfig::to_damper))
/// and held by `Tracker` for the lifetime of a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Damper {
    /// Tokens above which damping applies. `0` disables damping entirely.
    pub ceiling: usize,
}

impl Default for Damper {
    fn default() -> Self {
        Self::new(DEFAULT_DAMPING_CEILING)
    }
}

impl Damper {
    /// Build a damper with an explicit ceiling. `0` disables damping.
    pub fn new(ceiling: usize) -> Self {
        Self { ceiling }
    }

    /// Build a pass-through damper: every input reports its raw value.
    ///
    /// Identical to `new(0)`, named for what it means. `Tracker::new_in_memory()`
    /// uses it so tests see raw arithmetic unless they opt into damping.
    /// `#[allow(dead_code)]` because that caller is `#[cfg(test)]`: production
    /// reaches the same state through `damping_ceiling = 0` in config.
    #[allow(dead_code)]
    pub fn disabled() -> Self {
        Self::new(0)
    }

    /// Map a raw input-token count to its effective (damped) count.
    ///
    /// Identity below the ceiling and when disabled; logarithmic above it.
    pub fn damp_tokens(&self, input_tokens: usize) -> usize {
        if self.ceiling == 0 || input_tokens <= self.ceiling {
            return input_tokens;
        }
        let ceiling = self.ceiling as f64;
        let damped = ceiling * (1.0 + (input_tokens as f64 / ceiling).ln());
        // Above the ceiling the curve is bounded by the identity branch, so the
        // rounded result always fits the usize that carried the raw input.
        damped.round() as usize
    }

    /// Tokens saved against the damped input, keeping the sign.
    ///
    /// Negative when a filter emitted more than the wrapped command did — a real
    /// regression that callers must be able to see rather than have clamped away.
    pub fn effective_saved_signed(&self, input_tokens: usize, output_tokens: usize) -> i64 {
        self.damp_tokens(input_tokens) as i64 - output_tokens as i64
    }

    /// Tokens saved against the damped input, clamped at 0 for unsigned counters.
    pub fn effective_saved_clamped(&self, input_tokens: usize, output_tokens: usize) -> usize {
        self.effective_saved_signed(input_tokens, output_tokens)
            .max(0) as usize
    }

    /// Savings rate against the damped input, as a percentage.
    ///
    /// Keeps the sign (see [`Self::effective_saved_signed`]) and reports 0.0 when
    /// there is no effective input to divide by.
    pub fn effective_savings_pct(&self, input_tokens: usize, output_tokens: usize) -> f64 {
        let effective_input = self.damp_tokens(input_tokens);
        if effective_input == 0 {
            return 0.0;
        }
        (self.effective_saved_signed(input_tokens, output_tokens) as f64 / effective_input as f64)
            * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_below_ceiling() {
        let damper = Damper::default();
        for raw in [0, 1, 42, 5_000, 12_499] {
            assert_eq!(
                damper.damp_tokens(raw),
                raw,
                "input below the ceiling must pass through untouched"
            );
        }
    }

    #[test]
    fn test_exact_boundary_is_continuous() {
        let damper = Damper::default();
        assert_eq!(
            damper.damp_tokens(DEFAULT_DAMPING_CEILING),
            DEFAULT_DAMPING_CEILING
        );
        // No cliff on the far side: one token past the ceiling still maps to
        // one token past it, because the curve's slope there is exactly 1.0.
        assert_eq!(
            damper.damp_tokens(DEFAULT_DAMPING_CEILING + 1),
            DEFAULT_DAMPING_CEILING + 1
        );
    }

    #[test]
    fn test_runaway_command_damps_to_expected_scale() {
        // The motivating case: a `rg` that emitted ~26M tokens.
        // 26_000_000 / 12_500 = 2080; ln(2080) = 7.6401232...
        // 12_500 * 8.6401232... = 108_001.54 -> 108_002.
        let damper = Damper::default();
        assert_eq!(damper.damp_tokens(26_000_000), 108_002);
        // Still the biggest entry in any table, but ~8.6x the ceiling
        // instead of ~2080x.
        assert!(damper.damp_tokens(26_000_000) > DEFAULT_DAMPING_CEILING * 8);
        assert!(damper.damp_tokens(26_000_000) < DEFAULT_DAMPING_CEILING * 9);
    }

    #[test]
    fn test_monotonic_across_range() {
        let damper = Damper::default();
        // Non-decreasing, not strictly increasing: above the ceiling the slope
        // is C/I < 1, so adjacent raw inputs legitimately share a damped value.
        // What must never happen is a bigger raw input damping to a smaller one,
        // which would reorder the "top commands by tokens saved" table.
        let mut previous = 0usize;
        for raw in (0..=50_000_000).step_by(9_973) {
            let damped = damper.damp_tokens(raw);
            assert!(
                damped >= previous,
                "damping must never invert order: {raw} damped to {damped}, below previous {previous}"
            );
            assert!(
                damped <= raw,
                "damping must never inflate: {raw} damped to {damped}"
            );
            previous = damped;
        }
    }

    #[test]
    fn test_disabled_damper_is_pure_passthrough() {
        let damper = Damper::disabled();
        for raw in [0, 12_500, 26_000_000, usize::MAX] {
            assert_eq!(
                damper.damp_tokens(raw),
                raw,
                "a disabled damper must report raw values exactly"
            );
        }
    }

    #[test]
    fn test_custom_ceiling_scales_the_curve() {
        let damper = Damper::new(25_000);
        assert_eq!(damper.damp_tokens(25_000), 25_000);
        assert_eq!(damper.damp_tokens(20_000), 20_000);
        // 26_000_000 / 25_000 = 1040; ln(1040) = 6.9469...
        assert_eq!(damper.damp_tokens(26_000_000), 198_674);
    }

    #[test]
    fn test_savings_against_damped_input() {
        let damper = Damper::default();
        // Ordinary command, below the ceiling: raw arithmetic.
        assert_eq!(damper.effective_saved_signed(1_000, 200), 800);
        assert_eq!(damper.effective_saved_clamped(1_000, 200), 800);
        assert!((damper.effective_savings_pct(1_000, 200) - 80.0).abs() < 1e-9);

        // Runaway command: savings are measured against the damped input, so it
        // contributes 108_002 - 400 rather than 25_999_600.
        assert_eq!(damper.effective_saved_signed(26_000_000, 400), 107_602);
    }

    #[test]
    fn test_regression_keeps_its_negative_sign() {
        let damper = Damper::default();
        // A filter that emitted more than the wrapped command did.
        assert_eq!(damper.effective_saved_signed(100, 150), -50);
        assert_eq!(
            damper.effective_saved_clamped(100, 150),
            0,
            "unsigned counters clamp rather than wrapping to a huge usize"
        );
        assert!(
            damper.effective_savings_pct(100, 150) < 0.0,
            "a regression must stay visible as a negative rate, not a fake 0%"
        );

        // Damping can itself create a regression: a huge raw input whose output
        // exceeds the damped ceiling now reads as negative, which is the point —
        // it emitted more than Claude Code would ever have been shown.
        assert!(damper.effective_saved_signed(26_000_000, 200_000) < 0);
    }

    #[test]
    fn test_zero_input_reports_zero_rate() {
        // Passthrough rows record (0, 0) and must not dilute or divide by zero.
        assert!((Damper::default().effective_savings_pct(0, 0) - 0.0).abs() < f64::EPSILON);
        assert!((Damper::disabled().effective_savings_pct(0, 0) - 0.0).abs() < f64::EPSILON);
    }
}
