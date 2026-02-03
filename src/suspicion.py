"""
Suspicion models for OOD detection in adaptive inference.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors

class SuspicionModel:
    """Base class for suspicion models."""
    def __init__(self):
        self.is_dynamic = False

    def fit(self, X, y=None):
        pass

    def compute_suspicion(self, X):
        raise NotImplementedError

    def compute_ood_score(self, X):
        """Compute score for OOD detection metrics (higher is more OOD)."""
        return self.compute_suspicion(X)


class KNNSuspicion(SuspicionModel):
    """
    U3: External OOD suspicion using KNN distance.
    High accuracy, high compute cost.
    """
    def __init__(self, k=10, subsample=10000):
        super().__init__()
        self.k = k
        self.subsample = subsample
        self.knn = NearestNeighbors(n_neighbors=k)
        self.train_data = None
        self.scale_factor = 1.0

    def fit(self, X, y=None):
        if len(X) > self.subsample:
            idx = np.random.choice(len(X), self.subsample, replace=False)
            self.train_data = X[idx]
        else:
            self.train_data = X
        self.knn.fit(self.train_data)
        
        # Compute normalization factor: mean + 3*std of training distances
        dists, _ = self.knn.kneighbors(self.train_data)
        mean_dist = np.mean(dists)
        std_dist = np.std(dists)
        self.scale_factor = mean_dist + 3 * std_dist

    def compute_suspicion(self, X):
        # Returns mean distance to k nearest neighbors
        dists, _ = self.knn.kneighbors(X)
        # Normalize so that typical ID data is < 1.0
        return np.mean(dists, axis=1) / (self.scale_factor + 1e-10)


class TrajectorySuspicion(SuspicionModel):
    """
    U1: Trajectory-based suspicion using prediction dynamics.
    Zero overhead (uses existing prefix stats).
    
    Suspicion = 1.0 - Margin
    """
    def __init__(self):
        super().__init__()
        self.is_dynamic = True

    def fit(self, X, y=None):
        # No training needed
        pass

    def compute_suspicion(self, stats, t=None):
        """
        Compute suspicion from current prefix stats.
        
        Args:
            stats: dict containing 'margin'
            t: current depth (optional)
        """
        # Margin-based suspicion (lower margin -> more suspicious)
        # If margin is 0 (tie), suspicion is 1. If margin is 1 (unanimous), suspicion is 0.
        margin = stats['margin']
        suspicion = 1.0 - margin
        return suspicion
    
    def compute_ood_score(self, X):
        # Cannot compute static OOD score without running inference
        return None