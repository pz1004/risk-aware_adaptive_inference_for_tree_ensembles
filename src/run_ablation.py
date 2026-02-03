"""
Ablation study for Risk-Aware Adaptive Inference hyperparameters.

Analyzes sensitivity to:
- hard_gate_threshold: OOD safety vs ID work reduction
- delta_id: ID disagreement vs ID work reduction
- delta_suspicious: OOD disagreement vs OOD work reduction
"""
import os
import sys
import json
import logging
import time
import numpy as np
from datetime import datetime
from itertools import product

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loaders import get_dataset
from models import train_random_forest, train_gradient_boosting, PrefixEvaluatorRF, PrefixEvaluatorGBM
from calibration import FlipRiskCalibrator, compute_calibration_metrics
from suspicion import KNNSuspicion
from stopping_policy import RiskAwareStoppingPolicy, AdaptiveInferenceEngine


def setup_logging(name='ablation'):
    """Setup logging for ablation study."""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(funcName)s | %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f'{name}.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def run_single_config(dataset_name, model_type, config, n_runs=3, seed=42):
    """
    Run experiment with a single hyperparameter configuration.

    Args:
        dataset_name: 'mnist', 'covertype', or 'higgs'
        model_type: 'rf' or 'gbm'
        config: dict with hyperparameters
        n_runs: number of runs
        seed: random seed

    Returns:
        dict of aggregated results
    """
    logger = logging.getLogger(__name__)

    results = {
        'id_test': {'accuracy': [], 'disagreement_rate': [], 'work_reduction': []},
        'near_ood': {'disagreement_rate': [], 'work_reduction': [], 'early_exit_error_rate': []},
        'far_ood': {'disagreement_rate': [], 'work_reduction': [], 'early_exit_error_rate': []}
    }

    for run in range(n_runs):
        run_seed = seed + run
        np.random.seed(run_seed)

        # Load data
        data = get_dataset(dataset_name, seed=run_seed)

        # Split training data for calibration
        n_train = len(data['X_train'])
        n_cal = min(5000, n_train // 4)
        cal_idx = np.random.choice(n_train, n_cal, replace=False)
        train_idx = np.array([i for i in range(n_train) if i not in cal_idx])

        X_train = data['X_train'][train_idx]
        y_train = data['y_train'][train_idx]
        X_cal = data['X_train'][cal_idx]

        # Train model
        if model_type == 'rf':
            model = train_random_forest(X_train, y_train, n_estimators=100, random_state=run_seed)
            prefix_evaluator = PrefixEvaluatorRF(model)
        else:
            model = train_gradient_boosting(X_train, y_train, n_estimators=100, random_state=run_seed)
            prefix_evaluator = PrefixEvaluatorGBM(model)

        T = prefix_evaluator.n_estimators

        # Build and fit calibrator
        calibrator = FlipRiskCalibrator(method='logistic')
        cal_features, cal_labels = calibrator.build_calibration_dataset(
            prefix_evaluator, X_cal,
            prefix_grid=[1, 2, 4, 8, 16, 32, 64, 100]
        )
        calibrator.fit(cal_features, cal_labels)

        # Setup suspicion model
        suspicion_model = KNNSuspicion(k=config.get('knn_k', 10))
        suspicion_model.fit(X_train)

        # Setup stopping policy with config hyperparameters
        stopping_policy = RiskAwareStoppingPolicy(
            calibrator=calibrator,
            suspicion_model=suspicion_model,
            delta_id=config['delta_id'],
            delta_suspicious=config['delta_suspicious'],
            suspicion_threshold=config.get('suspicion_threshold', 0.5),
            t_min_id=config.get('t_min_id', 0.1),
            t_min_suspicious=config.get('t_min_suspicious', 0.3),
            use_hard_gate=True,
            hard_gate_threshold=config['hard_gate_threshold']
        )

        # Create inference engine
        engine = AdaptiveInferenceEngine(
            prefix_evaluator=prefix_evaluator,
            calibrator=calibrator,
            stopping_policy=stopping_policy,
            suspicion_model=suspicion_model
        )

        # Evaluate on ID test
        id_preds, id_stops = engine.run(data['X_id_test'])
        full_preds = prefix_evaluator.get_full_predictions(data['X_id_test'])

        results['id_test']['accuracy'].append(np.mean(id_preds == data['y_id_test']))
        results['id_test']['disagreement_rate'].append(np.mean(id_preds != full_preds))
        results['id_test']['work_reduction'].append(1 - np.mean(id_stops) / T)

        # Evaluate on Near OOD
        near_preds, near_stops = engine.run(data['X_near_ood'])
        near_full_preds = prefix_evaluator.get_full_predictions(data['X_near_ood'])

        results['near_ood']['disagreement_rate'].append(np.mean(near_preds != near_full_preds))
        results['near_ood']['work_reduction'].append(1 - np.mean(near_stops) / T)

        early_mask = near_stops < T
        if np.any(early_mask):
            results['near_ood']['early_exit_error_rate'].append(
                np.mean(near_preds[early_mask] != near_full_preds[early_mask])
            )
        else:
            results['near_ood']['early_exit_error_rate'].append(0.0)

        # Evaluate on Far OOD
        far_preds, far_stops = engine.run(data['X_far_ood'])
        far_full_preds = prefix_evaluator.get_full_predictions(data['X_far_ood'])

        results['far_ood']['disagreement_rate'].append(np.mean(far_preds != far_full_preds))
        results['far_ood']['work_reduction'].append(1 - np.mean(far_stops) / T)

        early_mask = far_stops < T
        if np.any(early_mask):
            results['far_ood']['early_exit_error_rate'].append(
                np.mean(far_preds[early_mask] != far_full_preds[early_mask])
            )
        else:
            results['far_ood']['early_exit_error_rate'].append(0.0)

    # Aggregate results
    aggregated = {'config': config}
    for split in ['id_test', 'near_ood', 'far_ood']:
        aggregated[split] = {}
        for metric, values in results[split].items():
            aggregated[split][f'{metric}_mean'] = float(np.mean(values))
            aggregated[split][f'{metric}_std'] = float(np.std(values))

    return aggregated


def run_ablation_study(datasets=None, model_types=None, n_runs=3, seed=42,
                       ablation_type='full', part=None):
    """
    Run full ablation study across hyperparameter grid.

    Args:
        datasets: list of dataset names (default: all)
        model_types: list of model types (default: ['rf', 'gbm'])
        n_runs: number of runs per config
        seed: random seed
        ablation_type: 'full', 'hard_gate', 'delta', or 'quick'
        part: None (all), 1 (first half), or 2 (second half) for parallel execution

    Returns:
        dict of all results
    """
    part_suffix = f'_part{part}' if part else ''
    logger = setup_logging(f'ablation_{ablation_type}{part_suffix}')

    if datasets is None:
        datasets = ['covertype', 'higgs', 'mnist']
    if model_types is None:
        model_types = ['rf', 'gbm']

    # Define ablation grids
    if ablation_type == 'full':
        # Full grid (5 x 5 x 5 = 125 configs per dataset/model)
        grid = {
            'hard_gate_threshold': [0.5, 0.6, 0.7, 0.8, 0.9],
            'delta_id': [0.01, 0.03, 0.05, 0.07, 0.10],
            'delta_suspicious': [0.001, 0.005, 0.01, 0.02, 0.05]
        }
    elif ablation_type == 'hard_gate':
        # Focus on hard_gate_threshold only
        grid = {
            'hard_gate_threshold': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'delta_id': [0.05],
            'delta_suspicious': [0.01]
        }
    elif ablation_type == 'delta':
        # Focus on delta parameters
        grid = {
            'hard_gate_threshold': [0.6],
            'delta_id': [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15],
            'delta_suspicious': [0.001, 0.005, 0.01, 0.02, 0.05]
        }
    elif ablation_type == 'quick':
        # Quick test with minimal grid
        grid = {
            'hard_gate_threshold': [0.6, 0.8],
            'delta_id': [0.03, 0.05, 0.10],
            'delta_suspicious': [0.01]
        }
    else:
        raise ValueError(f"Unknown ablation_type: {ablation_type}")

    # Generate all config combinations
    keys = list(grid.keys())
    values = list(grid.values())
    all_configs = [dict(zip(keys, v)) for v in product(*values)]

    # Generate all dataset/model combinations
    all_combinations = [(d, m) for d in datasets for m in model_types]

    # Split workload if part is specified
    if part == 1:
        # Part 1: First half of dataset/model combinations
        mid = len(all_combinations) // 2
        selected_combinations = all_combinations[:mid] if mid > 0 else all_combinations[:1]
        configs = all_configs
        logger.info(f"PART 1: Running {len(selected_combinations)} dataset/model combinations")
        logger.info(f"  Combinations: {selected_combinations}")
    elif part == 2:
        # Part 2: Second half of dataset/model combinations
        mid = len(all_combinations) // 2
        selected_combinations = all_combinations[mid:]
        configs = all_configs
        logger.info(f"PART 2: Running {len(selected_combinations)} dataset/model combinations")
        logger.info(f"  Combinations: {selected_combinations}")
    else:
        # Run all
        selected_combinations = all_combinations
        configs = all_configs

    logger.info("=" * 60)
    logger.info(f"ABLATION STUDY: {ablation_type}" + (f" (Part {part})" if part else ""))
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Dataset/Model combinations: {selected_combinations}")
    logger.info(f"Configs per combination: {len(configs)}")
    logger.info(f"Runs per config: {n_runs}")
    logger.info(f"Total experiments: {len(selected_combinations) * len(configs) * n_runs}")
    logger.info("=" * 60)

    all_results = {
        'ablation_type': ablation_type,
        'part': part,
        'timestamp': datetime.now().isoformat(),
        'grid': grid,
        'n_runs': n_runs,
        'results': {}
    }

    total_configs = len(selected_combinations) * len(configs)
    current = 0

    for dataset, model_type in selected_combinations:
        key = f"{dataset}_{model_type}"
        all_results['results'][key] = []

        for config in configs:
                current += 1
                logger.info(f"\n[{current}/{total_configs}] {key} - {config}")

                start_time = time.time()
                try:
                    result = run_single_config(
                        dataset, model_type, config,
                        n_runs=n_runs, seed=seed
                    )
                    result['runtime'] = time.time() - start_time
                    all_results['results'][key].append(result)

                    # Log summary
                    logger.info(
                        f"  ID: work_red={result['id_test']['work_reduction_mean']:.2%}, "
                        f"disagree={result['id_test']['disagreement_rate_mean']:.2%}"
                    )
                    logger.info(
                        f"  OOD: far_disagree={result['far_ood']['disagreement_rate_mean']:.2%}, "
                        f"far_work_red={result['far_ood']['work_reduction_mean']:.2%}"
                    )
                except Exception as e:
                    logger.error(f"Failed: {e}")
                    import traceback
                    traceback.print_exc()

    # Save results
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(output_dir, exist_ok=True)
    part_suffix = f'_part{part}' if part else ''
    output_path = os.path.join(output_dir, f'ablation_{ablation_type}{part_suffix}.json')

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")

    return all_results


def merge_ablation_results(part1_path, part2_path, output_path=None):
    """
    Merge two ablation result files from parallel runs.

    Args:
        part1_path: path to part 1 results
        part2_path: path to part 2 results
        output_path: path for merged output (default: inferred from part1)
    """
    with open(part1_path) as f:
        part1 = json.load(f)
    with open(part2_path) as f:
        part2 = json.load(f)

    # Merge results
    merged = {
        'ablation_type': part1['ablation_type'],
        'part': 'merged',
        'timestamp': datetime.now().isoformat(),
        'grid': part1['grid'],
        'n_runs': part1['n_runs'],
        'results': {}
    }

    # Combine results from both parts
    merged['results'].update(part1['results'])
    merged['results'].update(part2['results'])

    # Determine output path
    if output_path is None:
        output_path = part1_path.replace('_part1', '_merged')

    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)

    print(f"Merged results saved to {output_path}")
    print(f"  Part 1 keys: {list(part1['results'].keys())}")
    print(f"  Part 2 keys: {list(part2['results'].keys())}")
    print(f"  Merged keys: {list(merged['results'].keys())}")

    return merged


def analyze_ablation_results(results_path):
    """
    Analyze ablation study results and print summary.

    Args:
        results_path: path to ablation results JSON
    """
    with open(results_path) as f:
        data = json.load(f)

    print("\n" + "=" * 80)
    print("ABLATION STUDY ANALYSIS")
    print("=" * 80)

    for key, results in data['results'].items():
        print(f"\n{'=' * 40}")
        print(f"{key.upper()}")
        print("=" * 40)

        # Find Pareto-optimal configs (maximize work reduction, minimize OOD disagreement)
        pareto_front = []
        for r in results:
            is_dominated = False
            for other in results:
                # other dominates r if it's better on both metrics
                if (other['id_test']['work_reduction_mean'] > r['id_test']['work_reduction_mean'] and
                    other['far_ood']['disagreement_rate_mean'] < r['far_ood']['disagreement_rate_mean']):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_front.append(r)

        print(f"\nPareto-optimal configurations ({len(pareto_front)}):")
        print("-" * 70)
        print(f"{'hard_gate':>10} {'delta_id':>10} {'delta_sus':>10} {'ID_work%':>10} {'OOD_dis%':>10}")
        print("-" * 70)

        # Sort by work reduction
        pareto_front.sort(key=lambda x: x['id_test']['work_reduction_mean'], reverse=True)

        for r in pareto_front[:10]:  # Top 10
            c = r['config']
            print(f"{c['hard_gate_threshold']:>10.2f} "
                  f"{c['delta_id']:>10.3f} "
                  f"{c['delta_suspicious']:>10.3f} "
                  f"{r['id_test']['work_reduction_mean']*100:>10.1f} "
                  f"{r['far_ood']['disagreement_rate_mean']*100:>10.2f}")

        # Best configs for different objectives
        print("\n" + "-" * 70)
        print("Best for specific objectives:")

        # Best ID work reduction with OOD disagree < 5%
        safe_configs = [r for r in results if r['far_ood']['disagreement_rate_mean'] < 0.05]
        if safe_configs:
            best_safe = max(safe_configs, key=lambda x: x['id_test']['work_reduction_mean'])
            print(f"\n  Best work reduction (OOD disagree < 5%):")
            print(f"    Config: {best_safe['config']}")
            print(f"    ID work reduction: {best_safe['id_test']['work_reduction_mean']:.1%}")
            print(f"    OOD disagreement: {best_safe['far_ood']['disagreement_rate_mean']:.2%}")

        # Best OOD safety with work reduction > 30%
        efficient_configs = [r for r in results if r['id_test']['work_reduction_mean'] > 0.30]
        if efficient_configs:
            best_efficient = min(efficient_configs, key=lambda x: x['far_ood']['disagreement_rate_mean'])
            print(f"\n  Best OOD safety (work reduction > 30%):")
            print(f"    Config: {best_efficient['config']}")
            print(f"    ID work reduction: {best_efficient['id_test']['work_reduction_mean']:.1%}")
            print(f"    OOD disagreement: {best_efficient['far_ood']['disagreement_rate_mean']:.2%}")


def plot_ablation_results(results_path, output_dir=None):
    """
    Generate plots for ablation study results.

    Args:
        results_path: path to ablation results JSON
        output_dir: directory to save plots (default: results/figures)
    """
    import matplotlib.pyplot as plt

    with open(results_path) as f:
        data = json.load(f)

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(results_path), 'figures')
    os.makedirs(output_dir, exist_ok=True)

    for key, results in data['results'].items():
        # Extract data
        work_reduction = [r['id_test']['work_reduction_mean'] for r in results]
        ood_disagree = [r['far_ood']['disagreement_rate_mean'] for r in results]
        hard_gate = [r['config']['hard_gate_threshold'] for r in results]
        delta_id = [r['config']['delta_id'] for r in results]

        # Plot 1: Pareto frontier (work reduction vs OOD disagreement)
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            [w * 100 for w in work_reduction],
            [d * 100 for d in ood_disagree],
            c=hard_gate, cmap='viridis', alpha=0.7, s=50
        )
        ax.set_xlabel('ID Work Reduction (%)', fontsize=12)
        ax.set_ylabel('Far OOD Disagreement (%)', fontsize=12)
        ax.set_title(f'{key}: Work Reduction vs OOD Safety', fontsize=14)
        ax.axhline(y=5, color='r', linestyle='--', alpha=0.5, label='5% threshold')
        ax.axvline(x=30, color='g', linestyle='--', alpha=0.5, label='30% threshold')
        plt.colorbar(scatter, label='hard_gate_threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{key}_pareto.png'), dpi=150)
        plt.close()

        # Plot 2: Sensitivity to hard_gate_threshold
        unique_hg = sorted(set(hard_gate))
        if len(unique_hg) > 1:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Group by hard_gate_threshold
            hg_work = {hg: [] for hg in unique_hg}
            hg_ood = {hg: [] for hg in unique_hg}
            for r in results:
                hg = r['config']['hard_gate_threshold']
                hg_work[hg].append(r['id_test']['work_reduction_mean'])
                hg_ood[hg].append(r['far_ood']['disagreement_rate_mean'])

            # Work reduction
            means = [np.mean(hg_work[hg]) * 100 for hg in unique_hg]
            stds = [np.std(hg_work[hg]) * 100 for hg in unique_hg]
            axes[0].errorbar(unique_hg, means, yerr=stds, marker='o', capsize=5)
            axes[0].set_xlabel('hard_gate_threshold', fontsize=12)
            axes[0].set_ylabel('ID Work Reduction (%)', fontsize=12)
            axes[0].set_title('Sensitivity: hard_gate_threshold → Work Reduction')
            axes[0].grid(True, alpha=0.3)

            # OOD disagreement
            means = [np.mean(hg_ood[hg]) * 100 for hg in unique_hg]
            stds = [np.std(hg_ood[hg]) * 100 for hg in unique_hg]
            axes[1].errorbar(unique_hg, means, yerr=stds, marker='o', capsize=5, color='orange')
            axes[1].set_xlabel('hard_gate_threshold', fontsize=12)
            axes[1].set_ylabel('Far OOD Disagreement (%)', fontsize=12)
            axes[1].set_title('Sensitivity: hard_gate_threshold → OOD Safety')
            axes[1].axhline(y=5, color='r', linestyle='--', alpha=0.5)
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{key}_hard_gate_sensitivity.png'), dpi=150)
            plt.close()

        # Plot 3: Sensitivity to delta_id
        unique_delta = sorted(set(delta_id))
        if len(unique_delta) > 1:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Group by delta_id
            delta_work = {d: [] for d in unique_delta}
            delta_disagree = {d: [] for d in unique_delta}
            for r in results:
                d = r['config']['delta_id']
                delta_work[d].append(r['id_test']['work_reduction_mean'])
                delta_disagree[d].append(r['id_test']['disagreement_rate_mean'])

            # Work reduction
            means = [np.mean(delta_work[d]) * 100 for d in unique_delta]
            stds = [np.std(delta_work[d]) * 100 for d in unique_delta]
            axes[0].errorbar(unique_delta, means, yerr=stds, marker='o', capsize=5)
            axes[0].set_xlabel('delta_id', fontsize=12)
            axes[0].set_ylabel('ID Work Reduction (%)', fontsize=12)
            axes[0].set_title('Sensitivity: delta_id → Work Reduction')
            axes[0].grid(True, alpha=0.3)

            # ID disagreement
            means = [np.mean(delta_disagree[d]) * 100 for d in unique_delta]
            stds = [np.std(delta_disagree[d]) * 100 for d in unique_delta]
            axes[1].errorbar(unique_delta, means, yerr=stds, marker='o', capsize=5, color='orange')
            axes[1].set_xlabel('delta_id', fontsize=12)
            axes[1].set_ylabel('ID Disagreement (%)', fontsize=12)
            axes[1].set_title('Sensitivity: delta_id → ID Disagreement')
            axes[1].axhline(y=5, color='r', linestyle='--', alpha=0.5)
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{key}_delta_id_sensitivity.png'), dpi=150)
            plt.close()

    print(f"Plots saved to {output_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run ablation study')
    parser.add_argument('--type', choices=['full', 'hard_gate', 'delta', 'quick'],
                        default='quick', help='Type of ablation study')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Datasets to test (default: all)')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Model types to test (default: rf, gbm)')
    parser.add_argument('--runs', type=int, default=3,
                        help='Number of runs per config')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--part', type=int, choices=[1, 2], default=None,
                        help='Run only part 1 or 2 for parallel execution on two PCs')
    parser.add_argument('--analyze', type=str, default=None,
                        help='Path to results JSON to analyze (skip running)')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots after analysis')
    parser.add_argument('--merge', nargs=2, metavar=('PART1', 'PART2'),
                        help='Merge two part result files: --merge part1.json part2.json')

    args = parser.parse_args()

    if args.merge:
        merge_ablation_results(args.merge[0], args.merge[1])
    elif args.analyze:
        analyze_ablation_results(args.analyze)
        if args.plot:
            plot_ablation_results(args.analyze)
    else:
        results = run_ablation_study(
            datasets=args.datasets,
            model_types=args.models,
            n_runs=args.runs,
            seed=args.seed,
            ablation_type=args.type,
            part=args.part
        )

        # Auto-analyze
        part_suffix = f'_part{args.part}' if args.part else ''
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'results',
            f'ablation_{args.type}{part_suffix}.json'
        )
        analyze_ablation_results(output_path)

        if args.plot:
            plot_ablation_results(output_path)
