"""
Flip-risk calibration for early-exit ensembles.
"""
import numpy as np
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)


class FlipRiskCalibrator:
    """
    Calibrates the probability that prefix prediction agrees with full prediction.
    p_agree(t, x) = P(y_t(x) = y_T(x) | features)
    """

    def __init__(self, method='logistic'):
        """
        Args:
            method: 'logistic' or 'isotonic'
        """
        self.method = method
        self.model = None
        self.feature_names = ['stability', 'margin', 'confidence', 'depth']

    def build_calibration_dataset(self, prefix_evaluator, X_cal, prefix_grid=None):
        """
        Build calibration dataset from validation data.

        Args:
            prefix_evaluator: PrefixEvaluatorRF or PrefixEvaluatorGBM
            X_cal: Calibration data (n_samples, n_features)
            prefix_grid: List of prefix depths to evaluate (default: log-spaced)

        Returns:
            features: (n_samples * len(prefix_grid), n_features)
            labels: (n_samples * len(prefix_grid),) - 1 if agree, 0 if flip
        """
        n_estimators = prefix_evaluator.n_estimators

        if prefix_grid is None:
            # Log-spaced grid for efficiency
            prefix_grid = [1, 2, 4, 8, 16, 32, 64]
            prefix_grid = [t for t in prefix_grid if t <= n_estimators]
            if n_estimators not in prefix_grid:
                prefix_grid.append(n_estimators)

        # Get full predictions
        full_preds = prefix_evaluator.get_full_predictions(X_cal)

        all_features = []
        all_labels = []

        for t in prefix_grid:
            stats = prefix_evaluator.compute_prefix_stats(X_cal, t)
            prefix_preds = stats['predictions']

            # Labels: 1 if prefix agrees with full, 0 if flip
            agrees = (prefix_preds == full_preds).astype(int)

            # Features for calibration
            features = np.column_stack([
                stats.get('confidence', np.zeros(len(X_cal))),  # stability proxy
                stats['margin'],
                stats['confidence'],
                np.full(len(X_cal), stats['normalized_depth'])
            ])

            all_features.append(features)
            all_labels.append(agrees)

        features = np.vstack(all_features)
        labels = np.concatenate(all_labels)

        logger.info(f"Built calibration dataset: {len(labels)} samples, "
                    f"agreement rate: {np.mean(labels):.3f}")

        return features, labels

    def fit(self, features, labels):
        """
        Fit the calibration model.

        Args:
            features: (n_samples, n_features)
            labels: (n_samples,) - 1 if agree, 0 if flip
        """
        logger.info(f"Fitting {self.method} calibrator...")

        if self.method == 'logistic':
            self.model = LogisticRegression(max_iter=1000)
            self.model.fit(features, labels)
        elif self.method == 'isotonic':
            # For isotonic, we use a single score (weighted combination)
            self.lr = LogisticRegression(max_iter=1000)
            self.lr.fit(features, labels)
            scores = self.lr.predict_proba(features)[:, 1]
            self.model = IsotonicRegression(out_of_bounds='clip')
            self.model.fit(scores, labels)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        logger.info("Calibrator fitted")

    def predict_agreement_prob(self, features):
        """
        Predict probability of agreement with full prediction.

        Args:
            features: (n_samples, n_features)

        Returns:
            p_agree: (n_samples,) probabilities
        """
        if self.method == 'logistic':
            return self.model.predict_proba(features)[:, 1]
        elif self.method == 'isotonic':
            scores = self.lr.predict_proba(features)[:, 1]
            return self.model.predict(scores)

    def predict_from_stats(self, stats):
        """
        Predict agreement probability from prefix stats dict.

        Args:
            stats: dict from compute_prefix_stats

        Returns:
            p_agree: (n_samples,) probabilities
        """
        n_samples = len(stats['confidence'])
        features = np.column_stack([
            stats.get('confidence', np.zeros(n_samples)),
            stats['margin'],
            stats['confidence'],
            np.full(n_samples, stats['normalized_depth'])
        ])
        return self.predict_agreement_prob(features)


def compute_calibration_metrics(y_true, y_prob, n_bins=10):
    """
    Compute calibration metrics.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities

    Returns:
        dict with ECE, MCE, Brier score, reliability data
    """
    # Brier score
    brier = np.mean((y_prob - y_true) ** 2)

    # Reliability diagram data
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    except ValueError:
        prob_true = np.array([np.mean(y_true)])
        prob_pred = np.array([np.mean(y_prob)])

    # ECE and MCE
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    ece = 0.0
    mce = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            bin_size = np.sum(mask) / len(y_true)
            bin_error = np.abs(bin_acc - bin_conf)
            ece += bin_size * bin_error
            mce = max(mce, bin_error)

    return {
        'ece': ece,
        'mce': mce,
        'brier': brier,
        'reliability_true': prob_true,
        'reliability_pred': prob_pred
    }


class StabilitySurrogateRF:
    """
    Compute stability surrogate for Random Forest.
    Uses Dirichlet posterior approximation.
    """

    def __init__(self, prior=1.0):
        self.prior = prior

    def compute(self, vote_counts, t, T):
        """
        Compute stability score.

        Args:
            vote_counts: (n_samples, n_classes)
            t: current trees evaluated
            T: total trees

        Returns:
            stability: (n_samples,) estimated probability leader remains
        """
        remaining = T - t
        n_classes = vote_counts.shape[1]

        # Get top two vote counts
        sorted_votes = np.sort(vote_counts, axis=1)[:, ::-1]
        lead = sorted_votes[:, 0]
        second = sorted_votes[:, 1]

        # Margin
        margin = lead - second

        # Simple approximation: probability that random remaining votes
        # won't flip the leader
        # Using normal approximation for binomial
        if remaining == 0:
            return np.ones(len(vote_counts))

        # Each remaining tree is like a multinomial trial
        # Approximate: remaining votes split uniformly among classes
        expected_gain_per_class = remaining / n_classes
        std_gain = np.sqrt(remaining * (1 / n_classes) * (1 - 1 / n_classes))

        # Probability margin stays positive
        # Need margin > remaining worst case (all remaining go to second)
        # P(margin + (lead_gain - second_gain) > 0)
        # Approximate as normal
        z = margin / (2 * std_gain + 1e-10)
        from scipy.stats import norm
        stability = norm.cdf(z)

        return stability


class StabilitySurrogateGBM:
    """
    Compute stability surrogate for Gradient Boosting.
    Uses logit margin and remaining capacity.
    """

    def __init__(self, max_stage_contrib=0.1):
        self.max_stage_contrib = max_stage_contrib

    def compute(self, logits, t, T, learning_rate=0.1):
        """
        Compute stability score.

        Args:
            logits: current logits (n_samples,) for binary or (n_samples, n_classes)
            t: current stages evaluated
            T: total stages

        Returns:
            stability: (n_samples,) estimated probability prediction won't flip
        """
        remaining = T - t

        if remaining == 0:
            if logits.ndim == 1:
                return np.ones(len(logits))
            else:
                return np.ones(logits.shape[0])

        # Maximum possible change in logits
        max_change = remaining * learning_rate * self.max_stage_contrib

        if logits.ndim == 1:
            # Binary case
            margin = np.abs(logits)
        else:
            # Multiclass case
            sorted_logits = np.sort(logits, axis=1)[:, ::-1]
            margin = sorted_logits[:, 0] - sorted_logits[:, 1]

        # Stability: probability margin exceeds max possible change
        # Using sigmoid-like function
        stability = 1 / (1 + np.exp(-5 * (margin - max_change) / (max_change + 1e-10)))

        return stability
