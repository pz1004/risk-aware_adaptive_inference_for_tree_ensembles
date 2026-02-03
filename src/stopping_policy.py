"""
Risk-aware stopping policy for adaptive inference.
"""
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)


class RiskAwareStoppingPolicy:
    """
    Risk-aware closed-loop stopping policy.

    Stop at first t satisfying:
    - p_agree(t, x) >= 1 - delta(x)
    - t >= t_min(x)

    Where delta(x) decreases with suspicion.
    """

    def __init__(self, calibrator, suspicion_model=None,
                 delta_id=0.05, delta_suspicious=0.01,
                 suspicion_threshold=0.5,
                 t_min_id=0.1, t_min_suspicious=0.3,
                 use_hard_gate=False, hard_gate_threshold=0.8):
        """
        Args:
            calibrator: FlipRiskCalibrator
            suspicion_model: suspicion score model (optional)
            delta_id: risk threshold for ID samples
            delta_suspicious: risk threshold for suspicious samples
            suspicion_threshold: threshold for considering sample suspicious
            t_min_id: minimum depth fraction for ID samples
            t_min_suspicious: minimum depth fraction for suspicious samples
            use_hard_gate: force full evaluation if suspicion > hard_gate_threshold
            hard_gate_threshold: threshold for hard gate
        """
        self.calibrator = calibrator
        self.suspicion_model = suspicion_model
        self.delta_id = delta_id
        self.delta_suspicious = delta_suspicious
        self.suspicion_threshold = suspicion_threshold
        self.t_min_id = t_min_id
        self.t_min_suspicious = t_min_suspicious
        self.use_hard_gate = use_hard_gate
        self.hard_gate_threshold = hard_gate_threshold

    def get_delta(self, suspicion):
        """Get risk threshold based on suspicion."""
        # Two-level policy
        is_suspicious = suspicion > self.suspicion_threshold
        delta = np.where(is_suspicious, self.delta_suspicious, self.delta_id)
        return delta

    def get_t_min(self, suspicion, T):
        """Get minimum depth based on suspicion."""
        is_suspicious = suspicion > self.suspicion_threshold
        t_min_frac = np.where(is_suspicious, self.t_min_suspicious, self.t_min_id)
        return (t_min_frac * T).astype(int)

    def should_stop(self, p_agree, suspicion, t, T):
        """
        Check if should stop at current depth.

        Args:
            p_agree: (n_samples,) agreement probabilities
            suspicion: (n_samples,) suspicion scores
            t: current depth
            T: total depth

        Returns:
            stop_mask: (n_samples,) boolean mask
        """
        # Hard gate: force full evaluation if very suspicious
        if self.use_hard_gate:
            force_full = suspicion > self.hard_gate_threshold
        else:
            force_full = np.zeros(len(suspicion), dtype=bool)

        # Get thresholds
        delta = self.get_delta(suspicion)
        t_min = self.get_t_min(suspicion, T)

        # Stop conditions
        agreement_met = p_agree >= (1 - delta)
        depth_met = t >= t_min
        not_forced_full = ~force_full

        stop_mask = agreement_met & depth_met & not_forced_full
        return stop_mask


class AdaptiveInferenceEngine:
    """
    Run adaptive inference with risk-aware stopping.
    """

    def __init__(self, prefix_evaluator, calibrator, stopping_policy,
                 suspicion_model=None, prefix_grid=None):
        """
        Args:
            prefix_evaluator: PrefixEvaluatorRF or PrefixEvaluatorGBM
            calibrator: FlipRiskCalibrator
            stopping_policy: RiskAwareStoppingPolicy
            suspicion_model: OOD suspicion model (optional)
            prefix_grid: depths to check (default: every step)
        """
        self.prefix_evaluator = prefix_evaluator
        self.calibrator = calibrator
        self.stopping_policy = stopping_policy
        self.suspicion_model = suspicion_model

        T = prefix_evaluator.n_estimators
        if prefix_grid is None:
            # Check at every step for accuracy
            self.prefix_grid = list(range(1, T + 1))
        else:
            self.prefix_grid = prefix_grid

    def run(self, X, return_trajectory=False):
        """
        Run adaptive inference.

        Args:
            X: Input samples (n_samples, n_features)
            return_trajectory: whether to return trajectory stats

        Returns:
            predictions: (n_samples,) final predictions
            stop_times: (n_samples,) stopping times
            trajectory: list of stats dicts (if return_trajectory)
        """
        n_samples = X.shape[0]
        T = self.prefix_evaluator.n_estimators

        # Initialize
        predictions = np.zeros(n_samples, dtype=int)
        stop_times = np.full(n_samples, T)
        stopped = np.zeros(n_samples, dtype=bool)

        # Compute suspicion if model available
        if self.suspicion_model is not None:
            if not self.suspicion_model.is_dynamic:
                suspicion = self.suspicion_model.compute_suspicion(X)
            else:
                suspicion = np.zeros(n_samples)
        else:
            suspicion = np.zeros(n_samples)

        trajectory = []
        stats_sequence = []

        # Initialize incremental state
        if hasattr(self.prefix_evaluator, 'initialize_state'):
            state = self.prefix_evaluator.initialize_state(X)
            incremental_mode = True
        else:
            incremental_mode = False

        start_time = time.time()

        for t in self.prefix_grid:
            if np.all(stopped):
                break

            # Compute prefix stats (incrementally if possible)
            active_indices = np.where(~stopped)[0]
            if incremental_mode:
                state = self.prefix_evaluator.update_state(state, X, t, active_indices)
                stats = self.prefix_evaluator.compute_stats_from_state(state, t)
            else:
                stats = self.prefix_evaluator.compute_prefix_stats(X, t)
            
            stats_sequence.append(stats)

            # Update suspicion if dynamic (U1: Trajectory-based)
            if self.suspicion_model is not None and self.suspicion_model.is_dynamic:
                # Only update for active samples to save compute, or vector update all
                suspicion = self.suspicion_model.compute_suspicion(stats, t)

            if return_trajectory:
                trajectory.append({
                    't': t,
                    'predictions': stats['predictions'].copy(),
                    'confidence': stats['confidence'].copy(),
                    'margin': stats['margin'].copy()
                })

            # Compute agreement probability
            p_agree = self.calibrator.predict_from_stats(stats)

            # Check stopping condition for non-stopped samples
            active = ~stopped
            if np.any(active):
                stop_mask = self.stopping_policy.should_stop(
                    p_agree[active], suspicion[active], t, T
                )

                # Update stopped samples
                stop_indices = np.where(active)[0][stop_mask]
                predictions[stop_indices] = stats['predictions'][stop_indices]
                stop_times[stop_indices] = t
                stopped[stop_indices] = True

        # Handle samples that never stopped (use full prediction)
        not_stopped = ~stopped
        if np.any(not_stopped):
            full_preds = self.prefix_evaluator.get_full_predictions(X[not_stopped])
            predictions[not_stopped] = full_preds
            stop_times[not_stopped] = T

        elapsed = time.time() - start_time
        mean_stop = np.mean(stop_times)
        work_reduction = 1 - mean_stop / T

        logger.info(f"Adaptive inference: mean_stop={mean_stop:.2f}/{T}, "
                    f"work_reduction={work_reduction:.2%}, time={elapsed:.3f}s")

        if return_trajectory:
            return predictions, stop_times, trajectory
        return predictions, stop_times


class ConstantThresholdPolicy:
    """
    Simple constant threshold stopping policy (baseline).
    """

    def __init__(self, calibrator, delta=0.05, t_min_frac=0.1):
        self.calibrator = calibrator
        self.delta = delta
        self.t_min_frac = t_min_frac

    def should_stop(self, p_agree, suspicion, t, T):
        """Check if should stop."""
        t_min = int(self.t_min_frac * T)
        stop_mask = (p_agree >= (1 - self.delta)) & (t >= t_min)
        return stop_mask


class UncalibratedLazyPolicy:
    """
    Uncalibrated lazy evaluation policy (baseline from prior work).
    """

    def __init__(self, alpha_stop=0.95, t_min=10):
        """
        Args:
            alpha_stop: confidence threshold for stopping
            t_min: minimum trees before stopping
        """
        self.alpha_stop = alpha_stop
        self.t_min = t_min

    def run(self, prefix_evaluator, X):
        """
        Run uncalibrated lazy inference.

        Args:
            prefix_evaluator: PrefixEvaluatorRF or PrefixEvaluatorGBM
            X: Input samples

        Returns:
            predictions, stop_times
        """
        n_samples = X.shape[0]
        T = prefix_evaluator.n_estimators

        predictions = np.zeros(n_samples, dtype=int)
        stop_times = np.full(n_samples, T)
        stopped = np.zeros(n_samples, dtype=bool)

        for t in range(1, T + 1):
            if np.all(stopped):
                break

            stats = prefix_evaluator.compute_prefix_stats(X, t)
            active = ~stopped

            if t >= self.t_min:
                # Stop if confidence exceeds threshold
                stop_mask = stats['confidence'][active] >= self.alpha_stop

                stop_indices = np.where(active)[0][stop_mask]
                predictions[stop_indices] = stats['predictions'][stop_indices]
                stop_times[stop_indices] = t
                stopped[stop_indices] = True

        # Full prediction for non-stopped
        not_stopped = ~stopped
        if np.any(not_stopped):
            full_preds = prefix_evaluator.get_full_predictions(X[not_stopped])
            predictions[not_stopped] = full_preds

        return predictions, stop_times
