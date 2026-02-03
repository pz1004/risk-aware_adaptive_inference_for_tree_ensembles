"""
Final experiments for journal paper.

Experiments:
1. Optimal configuration evaluation (10+ runs with best params)
2. Calibration quality analysis (reliability diagrams, ECE/MCE)
3. Runtime/speedup analysis
4. Statistical significance tests

Usage:
    python src/run_final_experiments.py --exp optimal --runs 10
    python src/run_final_experiments.py --exp calibration
    python src/run_final_experiments.py --exp runtime
    python src/run_final_experiments.py --exp significance
    python src/run_final_experiments.py --exp all --runs 10
"""
import sys
import os
import json
import logging
import argparse
from datetime import datetime
import time
import numpy as np
from scipy import stats

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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(f'logs/final_experiments_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = 'results/final'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Pareto-optimal configurations from ablation study (at 2% OOD threshold)
OPTIMAL_CONFIGS = {
    'covertype_rf': {'hard_gate_threshold': 0.9, 'delta_id': 0.03, 'delta_suspicious': 0.001},
    'covertype_gbm': {'hard_gate_threshold': 0.5, 'delta_id': 0.1, 'delta_suspicious': 0.001},
    'higgs_rf': {'hard_gate_threshold': 0.9, 'delta_id': 0.05, 'delta_suspicious': 0.005},
    'higgs_gbm': {'hard_gate_threshold': 0.9, 'delta_id': 0.1, 'delta_suspicious': 0.005},
    'mnist_rf': {'hard_gate_threshold': 0.7, 'delta_id': 0.1, 'delta_suspicious': 0.01},
    'mnist_gbm': {'hard_gate_threshold': 0.7, 'delta_id': 0.1, 'delta_suspicious': 0.05},
}


def get_dataset(dataset_name, seed=42):
    """Load dataset."""
    if dataset_name == 'mnist':
        return prepare_mnist_splits(seed=seed)
    elif dataset_name == 'covertype':
        return prepare_covertype_splits(seed=seed)
    elif dataset_name == 'higgs':
        return prepare_higgs_splits(seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def run_optimal_config_experiment(n_runs=10, seed=42):
    """
    EXP-1: Run proposed algorithm with optimal configurations.
    """
    logger.info("=" * 70)
    logger.info("EXP-1: OPTIMAL CONFIGURATION EVALUATION")
    logger.info("=" * 70)

    results = {}

    for combo_name, config in OPTIMAL_CONFIGS.items():
        dataset_name, model_type = combo_name.rsplit('_', 1)
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {combo_name} with optimal config: {config}")
        logger.info(f"{'='*50}")

        run_results = []

        for run_idx in range(n_runs):
            run_seed = seed + run_idx
            logger.info(f"  Run {run_idx + 1}/{n_runs} (seed={run_seed})")

            try:
                # Load data
                data = get_dataset(dataset_name, seed=run_seed)
                X_train = data['X_train']
                y_train = data['y_train']

                # Split for calibration
                n_cal = min(5000, len(X_train) // 4)
                rng = np.random.RandomState(run_seed)
                idx = rng.permutation(len(X_train))
                X_cal = X_train[idx[:n_cal]]
                X_train_main = X_train[idx[n_cal:]]
                y_train_main = y_train[idx[n_cal:]]

                # Train model
                if model_type == 'rf':
                    model = train_random_forest(X_train_main, y_train_main, n_estimators=100)
                    prefix_evaluator = PrefixEvaluatorRF(model)
                else:
                    model = train_gradient_boosting(X_train_main, y_train_main, n_estimators=100)
                    prefix_evaluator = PrefixEvaluatorGBM(model)

                # Fit calibrator
                calibrator = FlipRiskCalibrator()
                prefix_grid = [1, 2, 4, 8, 16, 32, 64, 100]
                prefix_grid = [t for t in prefix_grid if t <= prefix_evaluator.n_estimators]
                cal_features, cal_labels = calibrator.build_calibration_dataset(
                    prefix_evaluator, X_cal, prefix_grid=prefix_grid
                )
                calibrator.fit(cal_features, cal_labels)

                # Fit suspicion model
                suspicion_model = KNNSuspicion(k=10, subsample=10000)
                suspicion_model.fit(X_train_main)

                # Create policy with optimal config
                policy = RiskAwareStoppingPolicy(
                    calibrator=calibrator,
                    delta_id=config['delta_id'],
                    delta_suspicious=config['delta_suspicious'],
                    suspicion_threshold=0.5,
                    hard_gate_threshold=config['hard_gate_threshold'],
                    t_min_id=0.1,
                    t_min_suspicious=0.3,
                    use_hard_gate=True
                )

                engine = AdaptiveInferenceEngine(
                    prefix_evaluator=prefix_evaluator,
                    calibrator=calibrator,
                    stopping_policy=policy,
                    suspicion_model=suspicion_model
                )

                # Evaluate on all splits
                run_result = {'seed': run_seed}

                for split_name in ['id_test', 'near_ood', 'far_ood']:
                    X_key = 'X_' + split_name.replace('_test', '_test').replace('ood', 'ood')
                    if split_name == 'id_test':
                        X = data['X_id_test']
                        y_true = data['y_id_test']
                    elif split_name == 'near_ood':
                        X = data['X_near_ood']
                        y_true = data.get('y_near_ood', data['y_id_test'])
                    else:
                        X = data['X_far_ood']
                        y_true = data.get('y_far_ood', data['y_id_test'])

                    y_full = model.predict(X)
                    predictions, stop_times = engine.run(X)
                    predictions = np.array(predictions)
                    stop_times = np.array(stop_times)

                    T = prefix_evaluator.n_estimators
                    run_result[split_name] = {
                        'accuracy': float(np.mean(predictions == y_true)),
                        'disagreement_rate': float(np.mean(predictions != y_full)),
                        'work_reduction': float(1 - np.mean(stop_times) / T),
                        'mean_stop_time': float(np.mean(stop_times)),
                    }

                    # Early exit error rate
                    early_mask = stop_times < T
                    if np.sum(early_mask) > 0:
                        run_result[split_name]['early_exit_error_rate'] = float(
                            np.mean(predictions[early_mask] != y_full[early_mask])
                        )
                    else:
                        run_result[split_name]['early_exit_error_rate'] = 0.0

                # OOD detection AUROC using suspicion scores
                id_suspicion = suspicion_model.compute_suspicion(data['X_id_test'])
                near_suspicion = suspicion_model.compute_suspicion(data['X_near_ood'])
                far_suspicion = suspicion_model.compute_suspicion(data['X_far_ood'])

                from sklearn.metrics import roc_auc_score
                run_result['near_auroc'] = float(roc_auc_score(
                    np.concatenate([np.zeros(len(id_suspicion)), np.ones(len(near_suspicion))]),
                    np.concatenate([id_suspicion, near_suspicion])
                ))
                run_result['far_auroc'] = float(roc_auc_score(
                    np.concatenate([np.zeros(len(id_suspicion)), np.ones(len(far_suspicion))]),
                    np.concatenate([id_suspicion, far_suspicion])
                ))

                run_results.append(run_result)

            except Exception as e:
                logger.error(f"Error in run {run_idx + 1}: {e}")
                import traceback
                traceback.print_exc()

        # Aggregate results
        if run_results:
            aggregated = aggregate_run_results(run_results)
            results[combo_name] = {
                'config': config,
                'n_runs': len(run_results),
                'runs': run_results,
                'aggregated': aggregated
            }

    # Save results
    output_path = os.path.join(OUTPUT_DIR, 'optimal_config_results.json')
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'n_runs': n_runs,
            'results': results
        }, f, indent=2)
    logger.info(f"\nSaved: {output_path}")

    return results


def aggregate_run_results(runs):
    """Aggregate results across runs."""
    agg = {}

    # Aggregate per-split metrics
    for split in ['id_test', 'near_ood', 'far_ood']:
        agg[split] = {}
        for metric in ['accuracy', 'disagreement_rate', 'work_reduction', 'early_exit_error_rate']:
            values = [r[split][metric] for r in runs if metric in r[split]]
            if values:
                agg[split][f'{metric}_mean'] = float(np.mean(values))
                agg[split][f'{metric}_std'] = float(np.std(values))
                agg[split][f'{metric}_ci95'] = float(1.96 * np.std(values) / np.sqrt(len(values)))

    # Aggregate AUROC
    for metric in ['near_auroc', 'far_auroc']:
        values = [r[metric] for r in runs if metric in r]
        if values:
            agg[f'{metric}_mean'] = float(np.mean(values))
            agg[f'{metric}_std'] = float(np.std(values))

    return agg


def run_calibration_analysis(seed=42):
    """
    EXP-2: Calibration quality analysis.
    """
    logger.info("=" * 70)
    logger.info("EXP-2: CALIBRATION QUALITY ANALYSIS")
    logger.info("=" * 70)

    results = {}

    for dataset_name in ['covertype', 'higgs', 'mnist']:
        for model_type in ['rf', 'gbm']:
            combo_name = f"{dataset_name}_{model_type}"
            logger.info(f"\nAnalyzing {combo_name}...")

            data = get_dataset(dataset_name, seed=seed)
            X_train = data['X_train']
            y_train = data['y_train']

            # Split for calibration
            n_cal = min(5000, len(X_train) // 4)
            X_cal = X_train[-n_cal:]
            X_train_main = X_train[:-n_cal]
            y_train_main = y_train[:-n_cal]

            # Train model
            if model_type == 'rf':
                model = train_random_forest(X_train_main, y_train_main, n_estimators=100)
                prefix_evaluator = PrefixEvaluatorRF(model)
            else:
                model = train_gradient_boosting(X_train_main, y_train_main, n_estimators=100)
                prefix_evaluator = PrefixEvaluatorGBM(model)

            # Fit calibrator
            calibrator = FlipRiskCalibrator()
            prefix_grid = [1, 2, 4, 8, 16, 32, 64, 100]
            prefix_grid = [t for t in prefix_grid if t <= prefix_evaluator.n_estimators]
            cal_features, cal_labels = calibrator.build_calibration_dataset(
                prefix_evaluator, X_cal, prefix_grid=prefix_grid
            )
            calibrator.fit(cal_features, cal_labels)

            # Evaluate on test set
            X_test = data['X_id_test']
            y_full = model.predict(X_test)

            # Collect predictions and confidences at different depths
            reliability_data = {'bins': [], 'accuracy': [], 'confidence': [], 'count': []}

            for t in prefix_grid:
                # Incrementally build prefix stats using the shared evaluator API.
                state = prefix_evaluator.initialize_state(X_test)
                stats_batch = None
                for depth in range(1, t + 1):
                    state = prefix_evaluator.update_state(state, X_test, depth)
                    stats_batch = prefix_evaluator.compute_stats_from_state(state, depth)

                # Get calibrated probabilities
                p_agree = calibrator.predict_from_stats(stats_batch)

                # Actual agreement
                actual_agree = (stats_batch['predictions'] == y_full).astype(float)

                # Bin by predicted probability
                n_bins = 10
                bin_edges = np.linspace(0, 1, n_bins + 1)

                for i in range(n_bins):
                    mask = (p_agree >= bin_edges[i]) & (p_agree < bin_edges[i + 1])
                    if np.sum(mask) > 0:
                        reliability_data['bins'].append((bin_edges[i] + bin_edges[i + 1]) / 2)
                        reliability_data['accuracy'].append(float(np.mean(actual_agree[mask])))
                        reliability_data['confidence'].append(float(np.mean(p_agree[mask])))
                        reliability_data['count'].append(int(np.sum(mask)))

            # Compute ECE and MCE
            ece = 0.0
            mce = 0.0
            total = sum(reliability_data['count'])

            for acc, conf, count in zip(reliability_data['accuracy'],
                                         reliability_data['confidence'],
                                         reliability_data['count']):
                gap = abs(acc - conf)
                ece += (count / total) * gap
                mce = max(mce, gap)

            results[combo_name] = {
                'ece': float(ece),
                'mce': float(mce),
                'reliability_data': reliability_data
            }

            logger.info(f"  ECE: {ece:.4f}, MCE: {mce:.4f}")

    # Save results
    output_path = os.path.join(OUTPUT_DIR, 'calibration_analysis.json')
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2)
    logger.info(f"\nSaved: {output_path}")

    return results


def run_runtime_analysis(n_runs=5, seed=42):
    """
    EXP-3: Runtime/speedup analysis.
    """
    logger.info("=" * 70)
    logger.info("EXP-3: RUNTIME ANALYSIS")
    logger.info("=" * 70)

    results = {}

    for dataset_name in ['covertype', 'higgs', 'mnist']:
        for model_type in ['rf', 'gbm']:
            combo_name = f"{dataset_name}_{model_type}"
            config = OPTIMAL_CONFIGS[combo_name]
            logger.info(f"\nMeasuring runtime for {combo_name}...")

            runtime_results = []

            for run_idx in range(n_runs):
                run_seed = seed + run_idx

                data = get_dataset(dataset_name, seed=run_seed)
                X_train = data['X_train']
                y_train = data['y_train']
                X_test = data['X_id_test']

                # Split for calibration
                n_cal = min(5000, len(X_train) // 4)
                X_cal = X_train[-n_cal:]
                X_train_main = X_train[:-n_cal]
                y_train_main = y_train[:-n_cal]

                # Train model
                if model_type == 'rf':
                    model = train_random_forest(X_train_main, y_train_main, n_estimators=100)
                    prefix_evaluator = PrefixEvaluatorRF(model)
                else:
                    model = train_gradient_boosting(X_train_main, y_train_main, n_estimators=100)
                    prefix_evaluator = PrefixEvaluatorGBM(model)

                # Fit calibrator and suspicion
                calibrator = FlipRiskCalibrator()
                prefix_grid = [1, 2, 4, 8, 16, 32, 64, 100]
                prefix_grid = [t for t in prefix_grid if t <= prefix_evaluator.n_estimators]
                cal_features, cal_labels = calibrator.build_calibration_dataset(
                    prefix_evaluator, X_cal, prefix_grid=prefix_grid
                )
                calibrator.fit(cal_features, cal_labels)

                suspicion_model = KNNSuspicion(k=10, subsample=10000)
                suspicion_model.fit(X_train_main)

                # Create policy
                policy = RiskAwareStoppingPolicy(
                    calibrator=calibrator,
                    delta_id=config['delta_id'],
                    delta_suspicious=config['delta_suspicious'],
                    suspicion_threshold=0.5,
                    hard_gate_threshold=config['hard_gate_threshold'],
                    t_min_id=0.1,
                    t_min_suspicious=0.3,
                    use_hard_gate=True
                )

                engine = AdaptiveInferenceEngine(
                    prefix_evaluator=prefix_evaluator,
                    calibrator=calibrator,
                    stopping_policy=policy,
                    suspicion_model=suspicion_model
                )

                # Measure full inference time
                start = time.perf_counter()
                _ = model.predict(X_test)
                full_time = time.perf_counter() - start

                # Measure proposed method time
                start = time.perf_counter()
                predictions, stop_times = engine.run(X_test)
                proposed_time = time.perf_counter() - start

                work_reduction = 1 - np.mean(stop_times) / prefix_evaluator.n_estimators
                speedup = full_time / proposed_time if proposed_time > 0 else 1.0

                runtime_results.append({
                    'full_time': full_time,
                    'proposed_time': proposed_time,
                    'speedup': speedup,
                    'work_reduction': work_reduction,
                    'n_samples': len(X_test)
                })

                logger.info(f"  Run {run_idx + 1}: speedup={speedup:.2f}x, work_red={work_reduction*100:.1f}%")

            # Aggregate
            results[combo_name] = {
                'runs': runtime_results,
                'speedup_mean': float(np.mean([r['speedup'] for r in runtime_results])),
                'speedup_std': float(np.std([r['speedup'] for r in runtime_results])),
                'work_reduction_mean': float(np.mean([r['work_reduction'] for r in runtime_results])),
            }

    # Save results
    output_path = os.path.join(OUTPUT_DIR, 'runtime_analysis.json')
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2)
    logger.info(f"\nSaved: {output_path}")

    return results


def run_significance_tests():
    """
    EXP-4: Statistical significance tests comparing proposed vs baselines.
    """
    logger.info("=" * 70)
    logger.info("EXP-4: STATISTICAL SIGNIFICANCE TESTS")
    logger.info("=" * 70)

    # Load comprehensive baseline results
    try:
        with open('baselines/comprehensive_results.json') as f:
            baseline_data = json.load(f)
    except FileNotFoundError:
        logger.error("baselines/comprehensive_results.json not found. Run comprehensive baselines first.")
        return None

    results = {}

    for combo_name, combo_data in baseline_data['results'].items():
        if 'runs' not in combo_data or len(combo_data['runs']) < 2:
            continue

        logger.info(f"\n{combo_name}:")
        results[combo_name] = {}

        # Extract per-run metrics for each method
        methods = ['full_inference', 'uncalibrated_lazy', 'constant_threshold', 'proposed']

        for split in ['id_test', 'near_ood', 'far_ood']:
            results[combo_name][split] = {}

            # Get proposed method values
            proposed_work = [r['early_exit'][methods[3]][split]['work_reduction']
                           for r in combo_data['runs'] if 'early_exit' in r]
            proposed_disagree = [r['early_exit'][methods[3]][split]['disagreement_rate']
                                for r in combo_data['runs'] if 'early_exit' in r]

            for baseline in methods[:3]:
                if baseline == 'full_inference':
                    baseline_work = [0.0] * len(proposed_work)
                    baseline_disagree = [0.0] * len(proposed_disagree)
                else:
                    baseline_work = [r['early_exit'][baseline][split]['work_reduction']
                                    for r in combo_data['runs'] if 'early_exit' in r]
                    baseline_disagree = [r['early_exit'][baseline][split]['disagreement_rate']
                                        for r in combo_data['runs'] if 'early_exit' in r]

                # Paired t-test for work reduction
                if len(proposed_work) >= 2 and len(baseline_work) >= 2:
                    t_stat, p_value = stats.ttest_rel(proposed_work, baseline_work)
                    results[combo_name][split][f'vs_{baseline}_work'] = {
                        'proposed_mean': float(np.mean(proposed_work)),
                        'baseline_mean': float(np.mean(baseline_work)),
                        'diff': float(np.mean(proposed_work) - np.mean(baseline_work)),
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant_0.05': bool(p_value < 0.05),
                        'significant_0.01': bool(p_value < 0.01),
                    }

            logger.info(f"  {split}: proposed work_red={np.mean(proposed_work)*100:.1f}% vs "
                       f"constant_threshold={np.mean(baseline_work)*100:.1f}%")

    # Save results
    output_path = os.path.join(OUTPUT_DIR, 'significance_tests.json')
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2)
    logger.info(f"\nSaved: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Run final experiments for journal paper')
    parser.add_argument('--exp', type=str, required=True,
                        choices=['optimal', 'calibration', 'runtime', 'significance', 'all'],
                        help='Experiment to run')
    parser.add_argument('--runs', type=int, default=10,
                        help='Number of runs for experiments that need multiple runs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    if args.exp == 'optimal' or args.exp == 'all':
        run_optimal_config_experiment(n_runs=args.runs, seed=args.seed)

    if args.exp == 'calibration' or args.exp == 'all':
        run_calibration_analysis(seed=args.seed)

    if args.exp == 'runtime' or args.exp == 'all':
        run_runtime_analysis(n_runs=min(args.runs, 5), seed=args.seed)

    if args.exp == 'significance' or args.exp == 'all':
        run_significance_tests()

    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENTS COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
