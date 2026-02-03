"""
Comprehensive baseline experiments for paper.

Runs:
1. OOD detection baselines (MSP, Entropy, KNN, Mahalanobis, LOF, etc.)
2. Early-exit baselines (ConstantThreshold, UncalibratedLazy, FullInference)
3. All dataset/model combinations
"""
import sys
import os
import json
import logging
import argparse
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loaders import prepare_mnist_splits, prepare_covertype_splits, prepare_higgs_splits
from models import train_random_forest, train_gradient_boosting, PrefixEvaluatorRF, PrefixEvaluatorGBM
from calibration import FlipRiskCalibrator
from suspicion import KNNSuspicion
from stopping_policy import (
    RiskAwareStoppingPolicy,
    AdaptiveInferenceEngine,
    ConstantThresholdPolicy,
    UncalibratedLazyPolicy
)
from baselines import run_all_baselines

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(funcName)s | %(message)s',
    handlers=[
        logging.FileHandler(f'logs/comprehensive_baselines_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def evaluate_early_exit_policy(policy, prefix_evaluator, calibrator, suspicion_model, X, y_full_pred,
                               split_name, use_suspicion=True):
    """
    Evaluate an early-exit policy.

    Returns metrics: work_reduction, disagreement_rate, early_exit_error_rate
    """
    n_samples = len(X)
    T = prefix_evaluator.n_estimators

    if hasattr(policy, 'run'):
        predictions, stop_times = policy.run(prefix_evaluator, X)
    else:
        engine = AdaptiveInferenceEngine(
            prefix_evaluator=prefix_evaluator,
            calibrator=calibrator,
            stopping_policy=policy,
            suspicion_model=suspicion_model if use_suspicion else None
        )
        predictions, stop_times = engine.run(X)

    stop_times = np.asarray(stop_times)
    predictions = np.asarray(predictions)

    # Compute metrics
    mean_stop_time = np.mean(stop_times)
    work_reduction = 1.0 - mean_stop_time / T
    disagreement_rate = np.mean(predictions != y_full_pred)

    # Early exit error rate (among those that exited early)
    early_mask = stop_times < T
    if np.sum(early_mask) > 0:
        early_exit_error_rate = np.mean(predictions[early_mask] != y_full_pred[early_mask])
    else:
        early_exit_error_rate = 0.0

    return {
        'mean_stop_time': float(mean_stop_time),
        'work_reduction': float(work_reduction),
        'disagreement_rate': float(disagreement_rate),
        'early_exit_error_rate': float(early_exit_error_rate),
        'n_early_exits': int(np.sum(early_mask)),
        'n_samples': n_samples
    }


def run_early_exit_baselines(model, prefix_evaluator, calibrator, suspicion_model,
                              X_id_test, X_near_ood, X_far_ood,
                              y_id_full, y_near_full, y_far_full):
    """
    Run early-exit baseline comparisons.
    """
    results = {}

    # 1. Full inference (no early exit)
    logger.info("Evaluating Full Inference baseline...")
    results['full_inference'] = {
        'id_test': {'work_reduction': 0.0, 'disagreement_rate': 0.0, 'early_exit_error_rate': 0.0},
        'near_ood': {'work_reduction': 0.0, 'disagreement_rate': 0.0, 'early_exit_error_rate': 0.0},
        'far_ood': {'work_reduction': 0.0, 'disagreement_rate': 0.0, 'early_exit_error_rate': 0.0}
    }

    # 2. Uncalibrated Lazy Policy (confidence threshold)
    logger.info("Evaluating Uncalibrated Lazy Policy...")
    uncal_policy = UncalibratedLazyPolicy(alpha_stop=0.95, t_min=10)

    results['uncalibrated_lazy'] = {
        'id_test': evaluate_early_exit_policy(
            uncal_policy, prefix_evaluator, calibrator, None,
            X_id_test, y_id_full, 'id_test', use_suspicion=False
        ),
        'near_ood': evaluate_early_exit_policy(
            uncal_policy, prefix_evaluator, calibrator, None,
            X_near_ood, y_near_full, 'near_ood', use_suspicion=False
        ),
        'far_ood': evaluate_early_exit_policy(
            uncal_policy, prefix_evaluator, calibrator, None,
            X_far_ood, y_far_full, 'far_ood', use_suspicion=False
        )
    }

    # 3. Constant Threshold Policy (calibrated, no OOD awareness)
    logger.info("Evaluating Constant Threshold Policy...")
    t_min_frac = min(1.0, 10 / max(1, prefix_evaluator.n_estimators))
    const_policy = ConstantThresholdPolicy(calibrator=calibrator, delta=0.05, t_min_frac=t_min_frac)

    results['constant_threshold'] = {
        'id_test': evaluate_early_exit_policy(
            const_policy, prefix_evaluator, calibrator, None,
            X_id_test, y_id_full, 'id_test', use_suspicion=False
        ),
        'near_ood': evaluate_early_exit_policy(
            const_policy, prefix_evaluator, calibrator, None,
            X_near_ood, y_near_full, 'near_ood', use_suspicion=False
        ),
        'far_ood': evaluate_early_exit_policy(
            const_policy, prefix_evaluator, calibrator, None,
            X_far_ood, y_far_full, 'far_ood', use_suspicion=False
        )
    }

    # 4. Proposed: Risk-Aware Policy (for comparison reference)
    logger.info("Evaluating Risk-Aware Policy (proposed)...")
    proposed_policy = RiskAwareStoppingPolicy(
        calibrator=calibrator,
        delta_id=0.05,
        delta_suspicious=0.01,
        suspicion_threshold=0.5,
        hard_gate_threshold=0.6,
        t_min_id=0.1,
        t_min_suspicious=0.3,
        use_hard_gate=True
    )

    results['proposed'] = {
        'id_test': evaluate_early_exit_policy(
            proposed_policy, prefix_evaluator, calibrator, suspicion_model,
            X_id_test, y_id_full, 'id_test', use_suspicion=True
        ),
        'near_ood': evaluate_early_exit_policy(
            proposed_policy, prefix_evaluator, calibrator, suspicion_model,
            X_near_ood, y_near_full, 'near_ood', use_suspicion=True
        ),
        'far_ood': evaluate_early_exit_policy(
            proposed_policy, prefix_evaluator, calibrator, suspicion_model,
            X_far_ood, y_far_full, 'far_ood', use_suspicion=True
        )
    }

    return results


def run_single_experiment(dataset_name, model_type, seed=42, n_estimators=100):
    """Run a single experiment for one dataset/model combination."""
    logger.info(f"Running experiment: {dataset_name}_{model_type}, seed={seed}")
    if n_estimators < 1:
        raise ValueError(f"n_estimators must be >= 1, got {n_estimators}")

    # Load data
    if dataset_name == 'mnist':
        data = prepare_mnist_splits(seed=seed)
    elif dataset_name == 'covertype':
        data = prepare_covertype_splits(seed=seed)
    elif dataset_name == 'higgs':
        data = prepare_higgs_splits(seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    X_train = data['X_train']
    y_train = data['y_train']
    X_id_test = data['X_id_test']
    y_id_test = data['y_id_test']
    X_near_ood = data['X_near_ood']
    X_far_ood = data['X_far_ood']

    # Split training data for calibration
    n_train = len(X_train)
    if n_train < 2:
        raise ValueError(f"Need at least 2 training samples, got {n_train}")
    n_cal = min(5000, max(1, n_train // 4))
    if n_cal >= n_train:
        n_cal = n_train - 1

    rng = np.random.RandomState(seed)
    idx = rng.permutation(n_train)
    cal_idx = idx[:n_cal]
    train_idx = idx[n_cal:]

    X_cal = X_train[cal_idx]
    X_train_main = X_train[train_idx]
    y_train_main = y_train[train_idx]

    # Train model
    if model_type == 'rf':
        model = train_random_forest(X_train_main, y_train_main, n_estimators=n_estimators)
        prefix_evaluator = PrefixEvaluatorRF(model)
    else:
        model = train_gradient_boosting(X_train_main, y_train_main, n_estimators=n_estimators)
        prefix_evaluator = PrefixEvaluatorGBM(model)

    # Get full predictions
    y_id_full = model.predict(X_id_test)
    y_near_full = model.predict(X_near_ood)
    y_far_full = model.predict(X_far_ood)

    # Build and fit calibrator
    calibrator = FlipRiskCalibrator()
    prefix_grid = [1, 2, 4, 8, 16, 32, 64]
    prefix_grid = [t for t in prefix_grid if t <= prefix_evaluator.n_estimators]
    if prefix_evaluator.n_estimators not in prefix_grid:
        prefix_grid.append(prefix_evaluator.n_estimators)
    prefix_grid = sorted(prefix_grid)
    cal_features, cal_labels = calibrator.build_calibration_dataset(
        prefix_evaluator,
        X_cal,
        prefix_grid=prefix_grid
    )
    calibrator.fit(cal_features, cal_labels)

    # Fit suspicion model
    suspicion_model = KNNSuspicion(k=10, subsample=10000)
    suspicion_model.fit(X_train_main)

    # Run OOD detection baselines
    logger.info("Running OOD detection baselines...")
    ood_baselines_near = run_all_baselines(
        model, X_train_main, X_id_test, X_near_ood,
        model_type=model_type, subsample=5000
    )
    ood_baselines_far = run_all_baselines(
        model, X_train_main, X_id_test, X_far_ood,
        model_type=model_type, subsample=5000
    )

    # Run early-exit baselines
    logger.info("Running early-exit baselines...")
    early_exit_results = run_early_exit_baselines(
        model, prefix_evaluator, calibrator, suspicion_model,
        X_id_test, X_near_ood, X_far_ood,
        y_id_full, y_near_full, y_far_full
    )

    # Compute ID accuracy
    id_accuracy = float(np.mean(y_id_full == y_id_test))

    return {
        'dataset': dataset_name,
        'model_type': model_type,
        'seed': seed,
        'id_accuracy': id_accuracy,
        'ood_detection': {
            'near_ood': ood_baselines_near,
            'far_ood': ood_baselines_far
        },
        'early_exit': early_exit_results
    }


def run_comprehensive_baselines(datasets=None, model_types=None, n_runs=5, seed=42, n_estimators=100):
    """
    Run comprehensive baseline experiments.
    """
    if datasets is None:
        datasets = ['covertype', 'higgs', 'mnist']
    if model_types is None:
        model_types = ['rf', 'gbm']

    all_results = {}

    for dataset in datasets:
        for model_type in model_types:
            combo_name = f"{dataset}_{model_type}"
            logger.info(f"\n{'='*60}")
            logger.info(f"RUNNING: {combo_name}")
            logger.info(f"{'='*60}")

            combo_results = {
                'dataset': dataset,
                'model_type': model_type,
                'n_runs': n_runs,
                'runs': []
            }

            for run_idx in range(n_runs):
                run_seed = seed + run_idx
                logger.info(f"\n--- Run {run_idx + 1}/{n_runs} (seed={run_seed}) ---")

                try:
                    result = run_single_experiment(
                        dataset,
                        model_type,
                        seed=run_seed,
                        n_estimators=n_estimators
                    )
                    combo_results['runs'].append(result)
                except Exception as e:
                    logger.error(f"Error in run {run_idx + 1}: {e}")
                    import traceback
                    traceback.print_exc()

            # Aggregate results
            if combo_results['runs']:
                combo_results['aggregated'] = aggregate_results(combo_results['runs'])

            all_results[combo_name] = combo_results

    return all_results


def aggregate_results(runs):
    """Aggregate results across multiple runs."""
    aggregated = {
        'id_accuracy_mean': np.mean([r['id_accuracy'] for r in runs]),
        'id_accuracy_std': np.std([r['id_accuracy'] for r in runs]),
        'ood_detection': {},
        'early_exit': {}
    }

    # Aggregate OOD detection results
    for ood_type in ['near_ood', 'far_ood']:
        aggregated['ood_detection'][ood_type] = {}
        methods = runs[0]['ood_detection'][ood_type].keys()
        for method in methods:
            aurocs = [r['ood_detection'][ood_type][method]['auroc'] for r in runs]
            fprs = [r['ood_detection'][ood_type][method]['fpr_at_95'] for r in runs]
            aggregated['ood_detection'][ood_type][method] = {
                'auroc_mean': float(np.mean(aurocs)),
                'auroc_std': float(np.std(aurocs)),
                'fpr_at_95_mean': float(np.mean(fprs)),
                'fpr_at_95_std': float(np.std(fprs))
            }

    # Aggregate early-exit results
    policies = runs[0]['early_exit'].keys()
    for policy in policies:
        aggregated['early_exit'][policy] = {}
        for split in ['id_test', 'near_ood', 'far_ood']:
            metrics = {}
            for metric in ['work_reduction', 'disagreement_rate', 'early_exit_error_rate']:
                values = [r['early_exit'][policy][split][metric] for r in runs]
                metrics[f'{metric}_mean'] = float(np.mean(values))
                metrics[f'{metric}_std'] = float(np.std(values))
            aggregated['early_exit'][policy][split] = metrics

    return aggregated


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive baseline experiments')
    parser.add_argument('--datasets', nargs='+', default=['covertype', 'higgs', 'mnist'],
                        help='Datasets to evaluate')
    parser.add_argument('--models', nargs='+', default=['rf', 'gbm'],
                        help='Model types to evaluate')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs per configuration')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    parser.add_argument('--n-estimators', type=int, default=100,
                        help='Number of estimators for RF/GBM models')
    parser.add_argument('--output', type=str, default='baselines/comprehensive_results.json',
                        help='Output file path')

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    logger.info("Starting comprehensive baseline experiments")
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Runs per config: {args.runs}")
    logger.info(f"Estimators: {args.n_estimators}")

    results = run_comprehensive_baselines(
        datasets=args.datasets,
        model_types=args.models,
        n_runs=args.runs,
        seed=args.seed,
        n_estimators=args.n_estimators
    )

    # Add metadata
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'datasets': args.datasets,
            'models': args.models,
            'n_runs': args.runs,
            'seed': args.seed,
            'n_estimators': args.n_estimators
        },
        'results': results
    }

    # Save results
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to {args.output}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for combo_name, combo_results in results.items():
        if 'aggregated' in combo_results:
            agg = combo_results['aggregated']
            print(f"\n{combo_name.upper()}")
            print(f"  ID Accuracy: {agg['id_accuracy_mean']:.3f} ± {agg['id_accuracy_std']:.3f}")

            print(f"\n  Early-Exit Comparison (ID Test):")
            for policy in ['full_inference', 'uncalibrated_lazy', 'constant_threshold', 'proposed']:
                if policy in agg['early_exit']:
                    ee = agg['early_exit'][policy]['id_test']
                    print(f"    {policy:25s}: work_red={ee['work_reduction_mean']*100:.1f}%, "
                          f"disagree={ee['disagreement_rate_mean']*100:.2f}%")

            print(f"\n  Early-Exit Comparison (Near OOD):")
            for policy in ['full_inference', 'uncalibrated_lazy', 'constant_threshold', 'proposed']:
                if policy in agg['early_exit']:
                    ee = agg['early_exit'][policy]['near_ood']
                    print(f"    {policy:25s}: work_red={ee['work_reduction_mean']*100:.1f}%, "
                          f"disagree={ee['disagreement_rate_mean']*100:.2f}%")


if __name__ == '__main__':
    main()
