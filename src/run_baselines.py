"""
Run baseline experiments and save results.
"""
import os
import sys
import json
import logging
import time
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loaders import get_dataset
from models import train_random_forest, train_gradient_boosting, train_deep_ensemble
from baselines import (
    MSPBaseline, EntropyBaseline, MarginBaseline,
    IsolationForestBaseline, KNNDistanceBaseline, MahalanobisBaseline,
    LOFBaseline, RFEnsembleVarianceBaseline, DeepEnsembleBaseline,
    compute_ood_metrics
)

# Setup logging
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(funcName)s | %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'baselines.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_baselines_for_dataset(dataset_name, model_type='rf', n_runs=3, seed=42):
    """
    Run all baselines for a dataset.

    Args:
        dataset_name: 'mnist', 'covertype', or 'higgs'
        model_type: 'rf' or 'gbm'
        n_runs: number of runs for averaging
        seed: random seed

    Returns:
        dict of results
    """
    logger.info(f"="*60)
    logger.info(f"Running baselines for {dataset_name} with {model_type}")
    logger.info(f"="*60)

    all_results = {
        'dataset': dataset_name,
        'model_type': model_type,
        'n_runs': n_runs,
        'near_ood': {},
        'far_ood': {},
        'id_accuracy': [],
        'run_times': []
    }

    for run in range(n_runs):
        run_seed = seed + run
        np.random.seed(run_seed)

        logger.info(f"\n--- Run {run + 1}/{n_runs} (seed={run_seed}) ---")
        start_time = time.time()

        # Load data
        data = get_dataset(dataset_name, seed=run_seed)

        # Train model
        if model_type == 'rf':
            model = train_random_forest(
                data['X_train'], data['y_train'],
                n_estimators=100, random_state=run_seed
            )
        else:
            model = train_gradient_boosting(
                data['X_train'], data['y_train'],
                n_estimators=100, random_state=run_seed
            )

        # ID accuracy
        id_preds = model.predict(data['X_id_test'])
        id_acc = np.mean(id_preds == data['y_id_test'])
        all_results['id_accuracy'].append(id_acc)
        logger.info(f"ID accuracy: {id_acc:.4f}")

        # Subsample for heavy baselines
        X_train_sub = data['X_train']
        if len(X_train_sub) > 5000:
            idx = np.random.choice(len(X_train_sub), 5000, replace=False)
            X_train_sub = X_train_sub[idx]

        # Run baselines for Near OOD
        logger.info("Evaluating Near OOD...")
        near_results = run_baseline_suite(
            model, X_train_sub, data['X_id_test'], data['X_near_ood'],
            model_type=model_type
        )
        for name, metrics in near_results.items():
            if name not in all_results['near_ood']:
                all_results['near_ood'][name] = {'auroc': [], 'fpr_at_95': []}
            all_results['near_ood'][name]['auroc'].append(metrics['auroc'])
            all_results['near_ood'][name]['fpr_at_95'].append(metrics['fpr_at_95'])

        # Run baselines for Far OOD
        logger.info("Evaluating Far OOD...")
        far_results = run_baseline_suite(
            model, X_train_sub, data['X_id_test'], data['X_far_ood'],
            model_type=model_type
        )
        for name, metrics in far_results.items():
            if name not in all_results['far_ood']:
                all_results['far_ood'][name] = {'auroc': [], 'fpr_at_95': []}
            all_results['far_ood'][name]['auroc'].append(metrics['auroc'])
            all_results['far_ood'][name]['fpr_at_95'].append(metrics['fpr_at_95'])

        run_time = time.time() - start_time
        all_results['run_times'].append(run_time)
        logger.info(f"Run completed in {run_time:.2f}s")

    # Compute statistics
    all_results['id_accuracy_mean'] = np.mean(all_results['id_accuracy'])
    all_results['id_accuracy_std'] = np.std(all_results['id_accuracy'])

    for ood_type in ['near_ood', 'far_ood']:
        for name in all_results[ood_type]:
            for metric in ['auroc', 'fpr_at_95']:
                values = all_results[ood_type][name][metric]
                all_results[ood_type][name][f'{metric}_mean'] = np.mean(values)
                all_results[ood_type][name][f'{metric}_std'] = np.std(values)

    return all_results


def run_baseline_suite(model, X_train, X_id_test, X_ood, model_type='rf'):
    """Run all baseline methods."""
    results = {}

    # MSP
    msp = MSPBaseline(model)
    id_scores = msp.compute_score(X_id_test)
    ood_scores = msp.compute_score(X_ood)
    results['msp'] = compute_ood_metrics(id_scores, ood_scores)
    logger.info(f"  MSP: AUROC={results['msp']['auroc']:.4f}, FPR@95={results['msp']['fpr_at_95']:.4f}")

    # Entropy
    entropy = EntropyBaseline(model)
    id_scores = entropy.compute_score(X_id_test)
    ood_scores = entropy.compute_score(X_ood)
    results['entropy'] = compute_ood_metrics(id_scores, ood_scores)
    logger.info(f"  Entropy: AUROC={results['entropy']['auroc']:.4f}")

    # Margin
    margin = MarginBaseline(model)
    id_scores = margin.compute_score(X_id_test)
    ood_scores = margin.compute_score(X_ood)
    results['margin'] = compute_ood_metrics(id_scores, ood_scores)
    logger.info(f"  Margin: AUROC={results['margin']['auroc']:.4f}")

    # Isolation Forest
    iso = IsolationForestBaseline()
    iso.fit(X_train)
    id_scores = iso.compute_score(X_id_test)
    ood_scores = iso.compute_score(X_ood)
    results['isolation_forest'] = compute_ood_metrics(id_scores, ood_scores)
    logger.info(f"  Isolation Forest: AUROC={results['isolation_forest']['auroc']:.4f}")

    # KNN Distance
    knn = KNNDistanceBaseline(k=10)
    knn.fit(X_train)
    id_scores = knn.compute_score(X_id_test)
    ood_scores = knn.compute_score(X_ood)
    results['knn_distance'] = compute_ood_metrics(id_scores, ood_scores)
    logger.info(f"  KNN Distance: AUROC={results['knn_distance']['auroc']:.4f}")

    # Mahalanobis
    try:
        maha = MahalanobisBaseline()
        maha.fit(X_train)
        id_scores = maha.compute_score(X_id_test)
        ood_scores = maha.compute_score(X_ood)
        results['mahalanobis'] = compute_ood_metrics(id_scores, ood_scores)
        logger.info(f"  Mahalanobis: AUROC={results['mahalanobis']['auroc']:.4f}")
    except Exception as e:
        logger.warning(f"  Mahalanobis failed: {e}")
        results['mahalanobis'] = {'auroc': 0.5, 'fpr_at_95': 1.0}

    # LOF
    lof = LOFBaseline(n_neighbors=35)
    lof.fit(X_train)
    id_scores = lof.compute_score(X_id_test)
    ood_scores = lof.compute_score(X_ood)
    results['lof'] = compute_ood_metrics(id_scores, ood_scores)
    logger.info(f"  LOF: AUROC={results['lof']['auroc']:.4f}")

    # RF Ensemble Variance (only for RF)
    if model_type == 'rf':
        rf_var = RFEnsembleVarianceBaseline(model)
        id_scores = rf_var.compute_score(X_id_test)
        ood_scores = rf_var.compute_score(X_ood)
        results['rf_ensemble_variance'] = compute_ood_metrics(id_scores, ood_scores)
        logger.info(f"  RF Ensemble Variance: AUROC={results['rf_ensemble_variance']['auroc']:.4f}")

    return results


def main():
    """Run all baseline experiments."""
    logger.info("="*60)
    logger.info("Starting Baseline Experiments")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("="*60)

    all_results = {}

    # Run for each dataset and model type
    for dataset in ['covertype', 'higgs', 'mnist']:
        for model_type in ['rf', 'gbm']:
            key = f"{dataset}_{model_type}"
            try:
                results = run_baselines_for_dataset(
                    dataset, model_type=model_type, n_runs=3, seed=42
                )
                all_results[key] = results
            except Exception as e:
                logger.error(f"Failed for {key}: {e}")
                import traceback
                traceback.print_exc()

    # Save results
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'baselines')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'baseline_results.json')

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("BASELINE RESULTS SUMMARY")
    logger.info("="*60)

    for key, results in all_results.items():
        logger.info(f"\n{key}:")
        logger.info(f"  ID Accuracy: {results['id_accuracy_mean']:.4f} +/- {results['id_accuracy_std']:.4f}")

        # Best near OOD
        if results['near_ood']:
            best_near = max(results['near_ood'].items(),
                          key=lambda x: x[1].get('auroc_mean', 0))
            logger.info(f"  Best Near OOD: {best_near[0]} (AUROC={best_near[1]['auroc_mean']:.4f})")

        # Best far OOD
        if results['far_ood']:
            best_far = max(results['far_ood'].items(),
                         key=lambda x: x[1].get('auroc_mean', 0))
            logger.info(f"  Best Far OOD: {best_far[0]} (AUROC={best_far[1]['auroc_mean']:.4f})")


if __name__ == '__main__':
    main()
