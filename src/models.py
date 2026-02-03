"""
Model training and prefix evaluation for Random Forest and Gradient Boosting.
"""
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import time

logger = logging.getLogger(__name__)


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42, **kwargs):
    """Train a Random Forest classifier."""
    logger.info(f"Training Random Forest with {n_estimators} trees...")
    start_time = time.time()

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        **kwargs
    )
    rf.fit(X_train, y_train)

    elapsed = time.time() - start_time
    logger.info(f"RF training completed in {elapsed:.2f}s")
    return rf


def train_gradient_boosting(X_train, y_train, n_estimators=100, random_state=42, **kwargs):
    """Train a Gradient Boosting classifier."""
    logger.info(f"Training Gradient Boosting with {n_estimators} stages...")
    start_time = time.time()

    # Subsample for faster training if dataset is large
    if len(X_train) > 10000:
        idx = np.random.RandomState(random_state).choice(len(X_train), 10000, replace=False)
        X_sub, y_sub = X_train[idx], y_train[idx]
    else:
        X_sub, y_sub = X_train, y_train

    gbm = GradientBoostingClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        **kwargs
    )
    gbm.fit(X_sub, y_sub)

    elapsed = time.time() - start_time
    logger.info(f"GBM training completed in {elapsed:.2f}s")
    return gbm


class PrefixEvaluatorRF:
    """
    Prefix evaluator for Random Forest.
    Allows incremental evaluation tree-by-tree.
    """

    def __init__(self, rf_model):
        self.rf = rf_model
        self.n_estimators = len(rf_model.estimators_)
        self.n_classes = rf_model.n_classes_
        self.classes_ = rf_model.classes_

    def evaluate_prefix(self, X, t):
        """
        Evaluate first t trees and return vote counts.

        Args:
            X: Input samples (n_samples, n_features)
            t: Number of trees to evaluate (1 to n_estimators)

        Returns:
            vote_counts: (n_samples, n_classes) vote counts
        """
        n_samples = X.shape[0]
        vote_counts = np.zeros((n_samples, self.n_classes))

        for i in range(min(t, self.n_estimators)):
            tree_preds = self.rf.estimators_[i].predict(X)
            for c_idx, c in enumerate(self.classes_):
                vote_counts[:, c_idx] += (tree_preds == c)

        return vote_counts

    def get_prefix_predictions(self, X, t):
        """Get predictions from first t trees."""
        vote_counts = self.evaluate_prefix(X, t)
        pred_indices = np.argmax(vote_counts, axis=1)
        return self.classes_[pred_indices]

    def initialize_state(self, X):
        """Initialize state for incremental evaluation."""
        n_samples = X.shape[0]
        return {
            'vote_counts': np.zeros((n_samples, self.n_classes)),
            'n_samples': n_samples
        }

    def update_state(self, state, X, t, active_indices=None):
        """
        Update state with tree t (1-based index).
        Only evaluates the specific tree t-1.
        """
        tree_idx = t - 1
        if tree_idx >= self.n_estimators:
            return state

        estimator = self.rf.estimators_[tree_idx]
        
        if active_indices is None:
            preds = estimator.predict(X)
            # Vectorized update
            for c_idx, c in enumerate(self.classes_):
                # This creates a boolean mask and adds it
                state['vote_counts'][:, c_idx] += (preds == c)
        else:
            # Update only active samples
            if len(active_indices) > 0:
                X_subset = X[active_indices]
                preds = estimator.predict(X_subset)
                for c_idx, c in enumerate(self.classes_):
                    state['vote_counts'][active_indices, c_idx] += (preds == c)
        
        return state

    def compute_stats_from_state(self, state, t):
        """Compute stats from current accumulated state."""
        vote_counts = state['vote_counts']
        return self._compute_stats_from_votes(vote_counts, t)

    def get_full_predictions(self, X):
        """Get predictions from all trees."""
        return self.rf.predict(X)

    def compute_prefix_stats(self, X, t):
        """
        Compute prefix statistics for calibration.
        """
        vote_counts = self.evaluate_prefix(X, t)
        return self._compute_stats_from_votes(vote_counts, t)

    def _compute_stats_from_votes(self, vote_counts, t):
        pred_indices = np.argmax(vote_counts, axis=1)
        predictions = self.classes_[pred_indices]

        # Sort vote counts to get top1 and top2
        sorted_votes = np.sort(vote_counts, axis=1)[:, ::-1]
        margin = sorted_votes[:, 0] - sorted_votes[:, 1]
        confidence = sorted_votes[:, 0] / t

        # Entropy of vote distribution
        probs = vote_counts / t
        probs = np.clip(probs, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log(probs), axis=1)

        return {
            'predictions': predictions,
            'vote_counts': vote_counts,
            'margin': margin,
            'confidence': confidence,
            'normalized_depth': t / self.n_estimators,
            'entropy': entropy
        }


class PrefixEvaluatorGBM:
    """
    Prefix evaluator for Gradient Boosting.
    Allows incremental evaluation stage-by-stage.
    """

    def __init__(self, gbm_model):
        self.gbm = gbm_model
        self.n_estimators = gbm_model.n_estimators
        self.learning_rate = gbm_model.learning_rate

        # Check if binary or multiclass
        if hasattr(gbm_model, 'n_classes_'):
            self.n_classes = gbm_model.n_classes_
        else:
            self.n_classes = 2
        self.is_binary = self.n_classes == 2

    def initialize_state(self, X):
        """Initialize state with base prediction."""
        n_samples = X.shape[0]
        
        # Get initial prediction (prior)
        if self.is_binary:
            probs = self.gbm.init_.predict_proba(X)[:, 1]
            probs = np.clip(probs, 1e-10, 1 - 1e-10)
            logits = np.log(probs / (1 - probs))
        else:
            probs = self.gbm.init_.predict_proba(X)
            probs = np.clip(probs, 1e-10, 1.0)
            logits = np.log(probs)
            
        return {
            'logits': logits,
            'n_samples': n_samples
        }

    def update_state(self, state, X, t, active_indices=None):
        """Update state with stage t (1-based index)."""
        stage_idx = t - 1
        if stage_idx >= self.n_estimators:
            return state
            
        if active_indices is None:
            indices = slice(None)
            X_subset = X
        else:
            indices = active_indices
            X_subset = X[active_indices]
            
        if len(X_subset) == 0:
            return state

        if self.is_binary:
            stage_pred = self.gbm.estimators_[stage_idx, 0].predict(X_subset)
            state['logits'][indices] += self.learning_rate * stage_pred
        else:
            for k in range(self.n_classes):
                stage_pred = self.gbm.estimators_[stage_idx, k].predict(X_subset)
                state['logits'][indices, k] += self.learning_rate * stage_pred
                
        return state

    def compute_stats_from_state(self, state, t):
        return self._compute_stats_from_logits(state['logits'], t)

    def evaluate_prefix(self, X, t):
        """
        Evaluate first t stages and return raw scores/logits.

        Args:
            X: Input samples (n_samples, n_features)
            t: Number of stages to evaluate (1 to n_estimators)

        Returns:
            logits: raw scores (n_samples,) for binary, (n_samples, n_classes) for multiclass
        """
        n_samples = X.shape[0]

        # Get initial prediction (prior)
        if self.is_binary:
            # Convert initial probability to log-odds
            probs = self.gbm.init_.predict_proba(X)[:, 1]
            probs = np.clip(probs, 1e-10, 1 - 1e-10)
            logits = np.log(probs / (1 - probs))
        else:
            # Convert initial probabilities to log-probabilities (logits for softmax)
            probs = self.gbm.init_.predict_proba(X)
            probs = np.clip(probs, 1e-10, 1.0)
            logits = np.log(probs)

        # Add contributions from each stage
        for i in range(min(t, self.n_estimators)):
            if self.is_binary:
                stage_pred = self.gbm.estimators_[i, 0].predict(X)
                logits += self.learning_rate * stage_pred
            else:
                for k in range(self.n_classes):
                    stage_pred = self.gbm.estimators_[i, k].predict(X)
                    logits[:, k] += self.learning_rate * stage_pred

        return logits

    def get_prefix_predictions(self, X, t):
        """Get predictions from first t stages."""
        logits = self.evaluate_prefix(X, t)
        if self.is_binary:
            return (logits > 0).astype(int)
        else:
            return np.argmax(logits, axis=1)

    def get_full_predictions(self, X):
        """Get predictions from all stages."""
        return self.gbm.predict(X)

    def compute_prefix_stats(self, X, t):
        """
        Compute prefix statistics for calibration.

        Returns:
            dict with:
                - predictions: current predictions
                - logits: raw scores
                - margin: logit margin
                - confidence: max probability
                - normalized_depth: t / T
        """
        logits = self.evaluate_prefix(X, t)
        return self._compute_stats_from_logits(logits, t)

    def _compute_stats_from_logits(self, logits, t):
        if self.is_binary:
            predictions = (logits > 0).astype(int)
            margin = np.abs(logits)
            prob = 1 / (1 + np.exp(-logits))
            confidence = np.maximum(prob, 1 - prob)
            # Entropy for binary
            p = np.stack([1 - prob, prob], axis=1)
            entropy = -np.sum(p * np.log(p + 1e-10), axis=1)
        else:
            predictions = np.argmax(logits, axis=1)
            sorted_logits = np.sort(logits, axis=1)[:, ::-1]
            margin = sorted_logits[:, 0] - sorted_logits[:, 1]
            # Softmax for confidence
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            confidence = np.max(probs, axis=1)
            # Entropy for multiclass
            entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)

        return {
            'predictions': predictions,
            'logits': logits,
            'margin': margin,
            'confidence': confidence,
            'normalized_depth': t / self.n_estimators,
            'entropy': entropy
        }


def train_deep_ensemble(X_train, y_train, n_members=5, hidden_sizes=(128, 64),
                        max_iter=20, random_state=42):
    """Train a deep ensemble of MLPs."""
    logger.info(f"Training deep ensemble with {n_members} members...")
    start_time = time.time()

    ensemble = []
    for i in range(n_members):
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_sizes,
            max_iter=max_iter,
            random_state=random_state + i
        )
        # Bootstrap sampling
        idx = np.random.RandomState(random_state + i).choice(
            len(X_train), len(X_train), replace=True
        )
        mlp.fit(X_train[idx], y_train[idx])
        ensemble.append(mlp)

    elapsed = time.time() - start_time
    logger.info(f"Deep ensemble training completed in {elapsed:.2f}s")
    return ensemble


def ensemble_predict_proba(ensemble, X):
    """Get mean predictions from ensemble."""
    probs = np.array([m.predict_proba(X) for m in ensemble])
    return np.mean(probs, axis=0)


def ensemble_variance(ensemble, X):
    """Get prediction variance from ensemble."""
    probs = np.array([m.predict_proba(X) for m in ensemble])
    return np.mean(np.var(probs, axis=0), axis=1)


def ensemble_entropy(ensemble, X):
    """Get entropy of mean predictions."""
    mean_probs = ensemble_predict_proba(ensemble, X)
    mean_probs = np.clip(mean_probs, 1e-10, 1.0)
    return -np.sum(mean_probs * np.log(mean_probs), axis=1)
