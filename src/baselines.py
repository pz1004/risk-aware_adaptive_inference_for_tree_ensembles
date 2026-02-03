"""
Baseline OOD detection methods.
"""
import numpy as np
import logging
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.covariance import EmpiricalCovariance
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy.special import softmax

logger = logging.getLogger(__name__)


class MSPBaseline:
    """Maximum Softmax Probability baseline."""

    def __init__(self, model):
        self.model = model

    def compute_score(self, X):
        """
        OOD score: negative of max softmax probability.
        Higher = more likely OOD.
        """
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X)
        else:
            # For models without predict_proba
            probs = np.eye(2)[self.model.predict(X)]
        return -np.max(probs, axis=1)


class EntropyBaseline:
    """Entropy of predictions baseline."""

    def __init__(self, model):
        self.model = model

    def compute_score(self, X):
        """OOD score: entropy of predictions."""
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X)
        else:
            probs = np.eye(2)[self.model.predict(X)]

        probs = np.clip(probs, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log(probs), axis=1)
        return entropy


class MarginBaseline:
    """Margin (top1 - top2) baseline."""

    def __init__(self, model):
        self.model = model

    def compute_score(self, X):
        """OOD score: negative margin."""
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X)
        else:
            probs = np.eye(2)[self.model.predict(X)]

        sorted_probs = np.sort(probs, axis=1)[:, ::-1]
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]
        return -margin


class IsolationForestBaseline:
    """Isolation Forest for OOD detection."""

    def __init__(self, contamination=0.1, random_state=42):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X_train):
        logger.info("Fitting Isolation Forest...")
        self.model.fit(X_train)

    def compute_score(self, X):
        """OOD score: negative of anomaly score (lower is more anomalous)."""
        return -self.model.score_samples(X)


class KNNDistanceBaseline:
    """KNN distance for OOD detection."""

    def __init__(self, k=10):
        self.k = k
        self.knn = None

    def fit(self, X_train):
        logger.info(f"Fitting KNN with k={self.k}...")
        self.knn = NearestNeighbors(n_neighbors=self.k, n_jobs=-1)
        self.knn.fit(X_train)

    def compute_score(self, X):
        """OOD score: mean distance to k nearest neighbors."""
        distances, _ = self.knn.kneighbors(X)
        return np.mean(distances, axis=1)


class MahalanobisBaseline:
    """Mahalanobis distance for OOD detection."""

    def __init__(self):
        self.cov = None

    def fit(self, X_train):
        logger.info("Fitting Mahalanobis...")
        self.cov = EmpiricalCovariance().fit(X_train)

    def compute_score(self, X):
        """OOD score: Mahalanobis distance."""
        return self.cov.mahalanobis(X)


class LOFBaseline:
    """Local Outlier Factor for OOD detection."""

    def __init__(self, n_neighbors=35):
        self.n_neighbors = n_neighbors
        self.lof = None

    def fit(self, X_train):
        logger.info(f"Fitting LOF with n_neighbors={self.n_neighbors}...")
        # Subsample for efficiency
        if len(X_train) > 5000:
            idx = np.random.choice(len(X_train), 5000, replace=False)
            X_sub = X_train[idx]
        else:
            X_sub = X_train

        self.lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            novelty=True,
            n_jobs=-1
        )
        self.lof.fit(X_sub)

    def compute_score(self, X):
        """OOD score: negative LOF score."""
        return -self.lof.score_samples(X)


class OneClassSVMBaseline:
    """One-Class SVM for OOD detection."""

    def __init__(self, nu=0.05, kernel='rbf'):
        self.nu = nu
        self.kernel = kernel
        self.svm = None

    def fit(self, X_train):
        logger.info(f"Fitting One-Class SVM with nu={self.nu}...")
        # Subsample for efficiency
        if len(X_train) > 3000:
            idx = np.random.choice(len(X_train), 3000, replace=False)
            X_sub = X_train[idx]
        else:
            X_sub = X_train

        self.svm = OneClassSVM(nu=self.nu, kernel=self.kernel)
        self.svm.fit(X_sub)

    def compute_score(self, X):
        """OOD score: negative decision function."""
        return -self.svm.decision_function(X)


class GMMBaseline:
    """Gaussian Mixture Model for OOD detection."""

    def __init__(self, n_components=10, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.gmm = None

    def fit(self, X_train):
        logger.info(f"Fitting GMM with {self.n_components} components...")
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            random_state=self.random_state
        )
        self.gmm.fit(X_train)

    def compute_score(self, X):
        """OOD score: negative log-likelihood."""
        return -self.gmm.score_samples(X)


class PCAReconstructionBaseline:
    """PCA reconstruction error for OOD detection."""

    def __init__(self, variance_ratio=0.95):
        self.variance_ratio = variance_ratio
        self.pca = None

    def fit(self, X_train):
        logger.info(f"Fitting PCA with variance_ratio={self.variance_ratio}...")
        self.pca = PCA(n_components=self.variance_ratio)
        self.pca.fit(X_train)

    def compute_score(self, X):
        """OOD score: reconstruction error."""
        X_transformed = self.pca.transform(X)
        X_reconstructed = self.pca.inverse_transform(X_transformed)
        error = np.mean((X - X_reconstructed) ** 2, axis=1)
        return error


class RFEnsembleVarianceBaseline:
    """Random Forest ensemble variance for OOD detection."""

    def __init__(self, rf_model):
        self.rf = rf_model

    def compute_score(self, X):
        """OOD score: variance of tree predictions."""
        n_samples = X.shape[0]
        n_classes = self.rf.n_classes_

        # Get predictions from each tree
        tree_preds = np.zeros((len(self.rf.estimators_), n_samples, n_classes))
        for i, tree in enumerate(self.rf.estimators_):
            preds = tree.predict(X)
            for j, p in enumerate(preds):
                class_idx = np.where(self.rf.classes_ == p)[0][0]
                tree_preds[i, j, class_idx] = 1

        # Compute variance across trees
        mean_preds = np.mean(tree_preds, axis=0)
        variance = np.mean(np.var(tree_preds, axis=0), axis=1)
        return variance


class DeepEnsembleBaseline:
    """Deep ensemble (MLP) for OOD detection."""

    def __init__(self, ensemble):
        self.ensemble = ensemble

    def compute_variance(self, X):
        """OOD score: prediction variance."""
        probs = np.array([m.predict_proba(X) for m in self.ensemble])
        variance = np.mean(np.var(probs, axis=0), axis=1)
        return variance

    def compute_entropy(self, X):
        """OOD score: entropy of mean predictions."""
        probs = np.array([m.predict_proba(X) for m in self.ensemble])
        mean_probs = np.mean(probs, axis=0)
        mean_probs = np.clip(mean_probs, 1e-10, 1.0)
        entropy = -np.sum(mean_probs * np.log(mean_probs), axis=1)
        return entropy

    def compute_mutual_information(self, X):
        """OOD score: mutual information."""
        probs = np.array([m.predict_proba(X) for m in self.ensemble])
        mean_probs = np.mean(probs, axis=0)

        # Total entropy
        mean_probs_clipped = np.clip(mean_probs, 1e-10, 1.0)
        total_entropy = -np.sum(mean_probs_clipped * np.log(mean_probs_clipped), axis=1)

        # Expected entropy
        probs_clipped = np.clip(probs, 1e-10, 1.0)
        member_entropies = -np.sum(probs_clipped * np.log(probs_clipped), axis=2)
        expected_entropy = np.mean(member_entropies, axis=0)

        # Mutual information
        mi = total_entropy - expected_entropy
        return mi


def compute_ood_metrics(id_scores, ood_scores):
    """
    Compute OOD detection metrics.

    Args:
        id_scores: OOD scores for ID samples (higher = more OOD)
        ood_scores: OOD scores for OOD samples

    Returns:
        dict with AUROC, FPR@95
    """
    # Labels: 0 for ID, 1 for OOD
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])

    # AUROC
    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        auroc = 0.5

    # FPR@95 (FPR when TPR = 95%)
    # Find threshold where 95% of OOD is detected
    ood_sorted = np.sort(ood_scores)
    threshold_idx = int(0.05 * len(ood_sorted))  # 5% of OOD below threshold
    threshold = ood_sorted[threshold_idx] if threshold_idx < len(ood_sorted) else ood_sorted[0]

    # FPR at this threshold
    fpr_at_95 = np.mean(id_scores >= threshold)

    return {
        'auroc': auroc,
        'fpr_at_95': fpr_at_95
    }


def run_all_baselines(model, X_train, X_id_test, X_ood, model_type='rf', subsample=10000):
    """
    Run all baseline OOD detection methods.

    Args:
        model: trained classifier
        X_train: training data
        X_id_test: ID test data
        X_ood: OOD test data
        model_type: 'rf' or 'gbm'
        subsample: subsample size for heavy baselines

    Returns:
        dict of results for each baseline
    """
    results = {}

    # Subsample training data for heavy methods
    if len(X_train) > subsample and subsample > 0:
        idx = np.random.choice(len(X_train), subsample, replace=False)
        X_train_sub = X_train[idx]
    else:
        X_train_sub = X_train

    # Model-based baselines
    logger.info("Running MSP baseline...")
    msp = MSPBaseline(model)
    id_scores = msp.compute_score(X_id_test)
    ood_scores = msp.compute_score(X_ood)
    results['msp'] = compute_ood_metrics(id_scores, ood_scores)

    logger.info("Running Entropy baseline...")
    entropy = EntropyBaseline(model)
    id_scores = entropy.compute_score(X_id_test)
    ood_scores = entropy.compute_score(X_ood)
    results['entropy'] = compute_ood_metrics(id_scores, ood_scores)

    logger.info("Running Margin baseline...")
    margin = MarginBaseline(model)
    id_scores = margin.compute_score(X_id_test)
    ood_scores = margin.compute_score(X_ood)
    results['margin'] = compute_ood_metrics(id_scores, ood_scores)

    # Isolation Forest
    logger.info("Running Isolation Forest baseline...")
    iso = IsolationForestBaseline()
    iso.fit(X_train_sub)
    id_scores = iso.compute_score(X_id_test)
    ood_scores = iso.compute_score(X_ood)
    results['isolation_forest'] = compute_ood_metrics(id_scores, ood_scores)

    # KNN Distance
    logger.info("Running KNN Distance baseline...")
    knn = KNNDistanceBaseline(k=10)
    knn.fit(X_train_sub)
    id_scores = knn.compute_score(X_id_test)
    ood_scores = knn.compute_score(X_ood)
    results['knn_distance'] = compute_ood_metrics(id_scores, ood_scores)

    # Mahalanobis
    logger.info("Running Mahalanobis baseline...")
    try:
        maha = MahalanobisBaseline()
        maha.fit(X_train_sub)
        id_scores = maha.compute_score(X_id_test)
        ood_scores = maha.compute_score(X_ood)
        results['mahalanobis'] = compute_ood_metrics(id_scores, ood_scores)
    except Exception as e:
        logger.warning(f"Mahalanobis failed: {e}")
        results['mahalanobis'] = {'auroc': 0.5, 'fpr_at_95': 1.0}

    # LOF
    logger.info("Running LOF baseline...")
    lof = LOFBaseline(n_neighbors=35)
    lof.fit(X_train_sub)
    id_scores = lof.compute_score(X_id_test)
    ood_scores = lof.compute_score(X_ood)
    results['lof'] = compute_ood_metrics(id_scores, ood_scores)

    # RF-specific baselines
    if model_type == 'rf':
        logger.info("Running RF Ensemble Variance baseline...")
        rf_var = RFEnsembleVarianceBaseline(model)
        id_scores = rf_var.compute_score(X_id_test)
        ood_scores = rf_var.compute_score(X_ood)
        results['rf_ensemble_variance'] = compute_ood_metrics(id_scores, ood_scores)

    return results
