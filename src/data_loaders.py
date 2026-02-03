"""
Data loaders for MNIST, Covertype, and HIGGS datasets with OOD shifts.
"""
import numpy as np
import pickle
import gzip
import os
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


def load_mnist():
    """Load MNIST dataset from raw files."""
    mnist_dir = os.path.join(DATA_DIR, 'mnist', 'MNIST', 'raw')

    def read_idx(filename):
        with gzip.open(filename, 'rb') as f:
            # Read magic number and dimensions
            magic = int.from_bytes(f.read(4), 'big')
            if magic == 2051:  # Images
                n_images = int.from_bytes(f.read(4), 'big')
                n_rows = int.from_bytes(f.read(4), 'big')
                n_cols = int.from_bytes(f.read(4), 'big')
                data = np.frombuffer(f.read(), dtype=np.uint8)
                return data.reshape(n_images, n_rows * n_cols)
            elif magic == 2049:  # Labels
                n_labels = int.from_bytes(f.read(4), 'big')
                return np.frombuffer(f.read(), dtype=np.uint8)

    # Try gzipped files first, then uncompressed
    train_images_path = os.path.join(mnist_dir, 'train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(mnist_dir, 'train-labels-idx1-ubyte.gz')
    test_images_path = os.path.join(mnist_dir, 't10k-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(mnist_dir, 't10k-labels-idx1-ubyte.gz')

    X_train = read_idx(train_images_path)
    y_train = read_idx(train_labels_path)
    X_test = read_idx(test_images_path)
    y_test = read_idx(test_labels_path)

    # Scale to [0, 1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    return X_train, y_train, X_test, y_test


def load_covertype():
    """Load Covertype dataset from pickle files."""
    samples_path = os.path.join(DATA_DIR, 'covertype', 'samples_py3')
    targets_path = os.path.join(DATA_DIR, 'covertype', 'targets_py3')

    import zlib

    try:
        # The files are zlib-compressed joblib pickle format
        import joblib
        import io

        with open(samples_path, 'rb') as f:
            compressed_data = f.read()
            decompressed_data = zlib.decompress(compressed_data)
            X = joblib.load(io.BytesIO(decompressed_data))

        with open(targets_path, 'rb') as f:
            compressed_data = f.read()
            decompressed_data = zlib.decompress(compressed_data)
            y = joblib.load(io.BytesIO(decompressed_data))
    except Exception as e:
        logger.warning(f"Failed to load covertype with joblib: {e}")
        # Fallback: try regular pickle with zlib
        try:
            with open(samples_path, 'rb') as f:
                compressed_data = f.read()
                decompressed_data = zlib.decompress(compressed_data)
                X = pickle.loads(decompressed_data)

            with open(targets_path, 'rb') as f:
                compressed_data = f.read()
                decompressed_data = zlib.decompress(compressed_data)
                y = pickle.loads(decompressed_data)
        except Exception:
            with open(samples_path, 'rb') as f:
                X = pickle.load(f)
            with open(targets_path, 'rb') as f:
                y = pickle.load(f)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    return X, y


def load_higgs(max_samples=30000):
    """Load HIGGS dataset from CSV."""
    higgs_path = os.path.join(DATA_DIR, 'higgs', 'HIGGS.csv')

    try:
        # HIGGS CSV: first column is label, rest are features
        data = np.loadtxt(higgs_path, delimiter=',', max_rows=max_samples)
        y = data[:, 0].astype(np.int32)
        X = data[:, 1:].astype(np.float32)
        return X, y
    except Exception as e:
        logger.warning(f"Failed to load HIGGS: {e}. Generating synthetic data.")
        np.random.seed(42)
        X = np.random.randn(max_samples, 28).astype(np.float32)
        y = np.random.randint(0, 2, max_samples).astype(np.int32)
        return X, y


def prepare_mnist_splits(n_train=20000, n_test=5000, n_ood=5000, seed=42):
    """
    Prepare MNIST splits with ID and OOD data.

    Returns:
        dict with keys: X_train, y_train, X_id_test, y_id_test,
                       X_far_ood (rotated), X_near_ood (rotated 90deg)
    """
    logger.info("Loading MNIST data...")
    X_train_full, y_train_full, X_test_full, y_test_full = load_mnist()

    np.random.seed(seed)

    # Sample train set
    train_idx = np.random.choice(len(X_train_full), n_train, replace=False)
    X_train = X_train_full[train_idx]
    y_train = y_train_full[train_idx]

    # Sample ID test set
    test_idx = np.random.choice(len(X_test_full), n_test, replace=False)
    X_id_test = X_test_full[test_idx]
    y_id_test = y_test_full[test_idx]

    # Near OOD: 90-degree rotation of test images
    X_near_ood = rotate_images_90(X_id_test)

    # Far OOD: Permuted pixels (simulating different distribution)
    perm = np.random.permutation(784)
    X_far_ood = X_id_test[:, perm]

    logger.info(f"MNIST splits: train={len(X_train)}, id_test={len(X_id_test)}, "
                f"near_ood={len(X_near_ood)}, far_ood={len(X_far_ood)}")

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_id_test': X_id_test, 'y_id_test': y_id_test,
        'X_near_ood': X_near_ood, 'y_near_ood': y_id_test,  # Same labels for comparison
        'X_far_ood': X_far_ood, 'y_far_ood': y_id_test,
        'dataset_name': 'mnist'
    }


def rotate_images_90(X, img_size=28):
    """Rotate flattened images by 90 degrees."""
    n_samples = X.shape[0]
    X_rotated = np.zeros_like(X)
    for i in range(n_samples):
        img = X[i].reshape(img_size, img_size)
        img_rot = np.rot90(img)
        X_rotated[i] = img_rot.flatten()
    return X_rotated


def prepare_covertype_splits(n_train=20000, n_test=5000, n_ood=5000,
                             near_ood_type='noise', noise_scale=0.5, seed=42):
    """
    Prepare Covertype splits.
    ID classes: 1, 2, 3
    OOD classes: 4, 5, 6, 7
    """
    logger.info("Loading Covertype data...")
    X, y = load_covertype()

    np.random.seed(seed)

    # ID classes (1, 2, 3) - convert to 0-indexed
    id_mask = np.isin(y, [1, 2, 3])
    X_id = X[id_mask]
    y_id = y[id_mask] - 1  # Convert to 0, 1, 2

    # Far OOD classes (4, 5, 6, 7)
    ood_mask = np.isin(y, [4, 5, 6, 7])
    X_ood = X[ood_mask]
    y_ood = y[ood_mask]

    # Split ID into train and test
    if len(X_id) < n_train + n_test:
        n_train = int(0.8 * len(X_id))
        n_test = len(X_id) - n_train

    idx = np.random.permutation(len(X_id))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:n_train + n_test]

    X_train = X_id[train_idx]
    y_train = y_id[train_idx]
    X_id_test = X_id[test_idx]
    y_id_test = y_id[test_idx]

    # Fit scaler on training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_id_test = scaler.transform(X_id_test)

    # Sample Far OOD
    ood_idx = np.random.choice(len(X_ood), min(n_ood, len(X_ood)), replace=False)
    X_far_ood = scaler.transform(X_ood[ood_idx])

    # Create Near OOD based on type
    if near_ood_type == 'noise':
        train_std = np.std(X_train, axis=0)
        noise = np.random.randn(*X_id_test.shape) * train_std * noise_scale
        X_near_ood = X_id_test + noise
    elif near_ood_type == 'permute':
        X_near_ood = np.zeros_like(X_id_test)
        for i in range(X_id_test.shape[1]):
            X_near_ood[:, i] = np.random.permutation(X_id_test[:, i])
    elif near_ood_type == 'missing':
        X_near_ood = X_id_test.copy()
        missing_rate = 0.1
        mask = np.random.random(X_near_ood.shape) < missing_rate
        medians = np.median(X_train, axis=0)
        X_near_ood[mask] = np.tile(medians, (len(X_near_ood), 1))[mask]
    else:
        X_near_ood = X_id_test.copy()

    logger.info(f"Covertype splits: train={len(X_train)}, id_test={len(X_id_test)}, "
                f"near_ood={len(X_near_ood)}, far_ood={len(X_far_ood)}")

    return {
        'X_train': X_train.astype(np.float32),
        'y_train': y_train,
        'X_id_test': X_id_test.astype(np.float32),
        'y_id_test': y_id_test,
        'X_near_ood': X_near_ood.astype(np.float32),
        'y_near_ood': y_id_test,
        'X_far_ood': X_far_ood.astype(np.float32),
        'y_far_ood': y_ood[ood_idx],
        'dataset_name': 'covertype',
        'scaler': scaler
    }


def prepare_higgs_splits(n_train=20000, n_test=5000, n_ood=5000,
                         near_ood_type='noise', noise_scale=0.5, seed=42):
    """
    Prepare HIGGS splits.
    Far OOD: feature-wise permutation
    Near OOD: noise, permute, missing, or subgroup
    """
    logger.info("Loading HIGGS data...")
    X, y = load_higgs(max_samples=n_train + n_test + n_ood)

    np.random.seed(seed)

    # Handle NaN values with median imputation
    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    # Split into train and test
    idx = np.random.permutation(len(X))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:n_train + n_test]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_id_test = X[test_idx]
    y_id_test = y[test_idx]

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_id_test = scaler.transform(X_id_test)

    # Far OOD: feature-wise permutation
    X_far_ood = np.zeros_like(X_id_test)
    for i in range(X_id_test.shape[1]):
        X_far_ood[:, i] = np.random.permutation(X_id_test[:, i])

    # Near OOD based on type
    if near_ood_type == 'noise':
        train_std = np.std(X_train, axis=0)
        noise = np.random.randn(*X_id_test.shape) * train_std * noise_scale
        X_near_ood = X_id_test + noise
    elif near_ood_type == 'permute':
        X_near_ood = np.zeros_like(X_id_test)
        for i in range(X_id_test.shape[1]):
            X_near_ood[:, i] = np.random.permutation(X_id_test[:, i])
    elif near_ood_type == 'missing':
        X_near_ood = X_id_test.copy()
        missing_rate = 0.1
        mask = np.random.random(X_near_ood.shape) < missing_rate
        medians = np.median(X_train, axis=0)
        X_near_ood[mask] = np.tile(medians, (len(X_near_ood), 1))[mask]
    else:
        X_near_ood = X_id_test.copy()

    logger.info(f"HIGGS splits: train={len(X_train)}, id_test={len(X_id_test)}, "
                f"near_ood={len(X_near_ood)}, far_ood={len(X_far_ood)}")

    return {
        'X_train': X_train.astype(np.float32),
        'y_train': y_train,
        'X_id_test': X_id_test.astype(np.float32),
        'y_id_test': y_id_test,
        'X_near_ood': X_near_ood.astype(np.float32),
        'y_near_ood': y_id_test,
        'X_far_ood': X_far_ood.astype(np.float32),
        'y_far_ood': y_id_test,
        'dataset_name': 'higgs',
        'scaler': scaler
    }


def get_dataset(name, seed=42, **kwargs):
    """Get dataset by name."""
    if name == 'mnist':
        return prepare_mnist_splits(seed=seed, **kwargs)
    elif name == 'covertype':
        return prepare_covertype_splits(seed=seed, **kwargs)
    elif name == 'higgs':
        return prepare_higgs_splits(seed=seed, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")
