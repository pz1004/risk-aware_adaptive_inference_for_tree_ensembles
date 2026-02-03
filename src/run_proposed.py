"""
Run proposed Risk-Aware Adaptive Inference algorithm.
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
from models import train_random_forest, train_gradient_boosting, PrefixEvaluatorRF, PrefixEvaluatorGBM
from calibration import FlipRiskCalibrator, compute_calibration_metrics
from suspicion import KNNSuspicion, TrajectorySuspicion
from stopping_policy import (
    RiskAwareStoppingPolicy, AdaptiveInferenceEngine,
    UncalibratedLazyPolicy, ConstantThresholdPolicy
)
from baselines import compute_ood_metrics


def setup_logging(iteration):
    """Setup logging for iteration."""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(funcName)s | %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f'iteration_{iteration}.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def run_proposed_algorithm(dataset_name, model_type='rf', n_runs=3, seed=42,
                          delta_id=0.05, delta_suspicious=0.01,
                          suspicion_threshold=0.5,
                          hard_gate_threshold=0.8,
                          use_calibration=True, use_suspicion=True,
                          suspicion_type='knn'):
    """
    Run proposed Risk-Aware Adaptive Inference algorithm.

    Args:
        dataset_name: 'mnist', 'covertype', or 'higgs'
        model_type: 'rf' or 'gbm'
        n_runs: number of runs
        seed: random seed
        delta_id: risk threshold for ID samples
        delta_suspicious: risk threshold for suspicious samples
        suspicion_threshold: threshold for suspicious classification
        use_calibration: whether to use calibrated stopping
        use_suspicion: whether to use OOD suspicion
        suspicion_type: 'knn' or 'trajectory'

    Returns:
        dict of results
    """
    logger = logging.getLogger(__name__)

    logger.info(f"="*60)
    logger.info(f"Running Proposed Algorithm for {dataset_name} with {model_type}")
    logger.info(f"Config: delta_id={delta_id}, delta_suspicious={delta_suspicious}")
    logger.info(f"        hard_gate_threshold={hard_gate_threshold}")
    logger.info(f"        use_calibration={use_calibration}, use_suspicion={use_suspicion}")
    logger.info(f"        suspicion_type={suspicion_type}")
    logger.info(f"="*60)

    all_results = {
        'dataset': dataset_name,
        'model_type': model_type,
        'n_runs': n_runs,
        'config': {
            'delta_id': delta_id,
            'delta_suspicious': delta_suspicious,
            'suspicion_threshold': suspicion_threshold,
            'hard_gate_threshold': hard_gate_threshold,
            'use_calibration': use_calibration,
            'use_suspicion': use_suspicion,
            'suspicion_type': suspicion_type
        },
        'id_test': {
            'accuracy': [],
            'disagreement_rate': [],
            'mean_stop_time': [],
            'work_reduction': []
        },
        'near_ood': {
            'accuracy': [],
            'disagreement_rate': [],
            'mean_stop_time': [],
            'work_reduction': [],
            'early_exit_error_rate': []
        },
        'far_ood': {
            'accuracy': [],
            'disagreement_rate': [],
            'mean_stop_time': [],
            'work_reduction': [],
            'early_exit_error_rate': []
        },
        'calibration': {
            'ece': [],
            'brier': []
        },
        'ood_detection': {
            'near_auroc': [],
            'far_auroc': []
        },
        'run_times': []
    }

    for run in range(n_runs):
        run_seed = seed + run
        np.random.seed(run_seed)

        logger.info(f"\n--- Run {run + 1}/{n_runs} (seed={run_seed}) ---")
        start_time = time.time()

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
        y_cal = data['y_train'][cal_idx]

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

        # Evaluate calibration
        cal_probs = calibrator.predict_agreement_prob(cal_features)
        cal_metrics = compute_calibration_metrics(cal_labels, cal_probs)
        all_results['calibration']['ece'].append(cal_metrics['ece'])
        all_results['calibration']['brier'].append(cal_metrics['brier'])
        logger.info(f"Calibration: ECE={cal_metrics['ece']:.4f}, Brier={cal_metrics['brier']:.4f}")

        # Setup suspicion model
        if use_suspicion:
            if suspicion_type == 'trajectory':
                suspicion_model = TrajectorySuspicion()
            else:
                suspicion_model = KNNSuspicion(k=10)
                suspicion_model.fit(X_train)
        else:
            suspicion_model = None

        # Setup stopping policy
        if use_calibration:
            stopping_policy = RiskAwareStoppingPolicy(
                calibrator=calibrator,
                suspicion_model=suspicion_model,
                delta_id=delta_id,
                delta_suspicious=delta_suspicious,
                suspicion_threshold=suspicion_threshold,
                t_min_id=0.1,
                t_min_suspicious=0.3,
                use_hard_gate=use_suspicion,
                hard_gate_threshold=hard_gate_threshold
            )
        else:
            stopping_policy = ConstantThresholdPolicy(
                calibrator=calibrator,
                delta=delta_id,
                t_min_frac=0.1
            )

        # Create inference engine
        engine = AdaptiveInferenceEngine(
            prefix_evaluator=prefix_evaluator,
            calibrator=calibrator,
            stopping_policy=stopping_policy,
            suspicion_model=suspicion_model
        )

        # Evaluate on ID test
        logger.info("Evaluating on ID test...")
        id_preds, id_stops = engine.run(data['X_id_test'])
        full_preds = prefix_evaluator.get_full_predictions(data['X_id_test'])

        id_acc = np.mean(id_preds == data['y_id_test'])
        id_disagree = np.mean(id_preds != full_preds)
        id_mean_stop = np.mean(id_stops)
        id_work_reduction = 1 - id_mean_stop / T

        all_results['id_test']['accuracy'].append(id_acc)
        all_results['id_test']['disagreement_rate'].append(id_disagree)
        all_results['id_test']['mean_stop_time'].append(id_mean_stop)
        all_results['id_test']['work_reduction'].append(id_work_reduction)

        logger.info(f"ID Test: Acc={id_acc:.4f}, Disagree={id_disagree:.4f}, "
                   f"MeanStop={id_mean_stop:.1f}/{T}, WorkRed={id_work_reduction:.2%}")

        # Evaluate on Near OOD
        logger.info("Evaluating on Near OOD...")
        near_preds, near_stops = engine.run(data['X_near_ood'])
        near_full_preds = prefix_evaluator.get_full_predictions(data['X_near_ood'])

        near_acc = np.mean(near_preds == data['y_near_ood'])
        near_disagree = np.mean(near_preds != near_full_preds)
        near_mean_stop = np.mean(near_stops)
        near_work_reduction = 1 - near_mean_stop / T

        # Early exit error rate: among early exits, how many disagreed with full?
        early_mask = near_stops < T
        if np.any(early_mask):
            near_early_error = np.mean(near_preds[early_mask] != near_full_preds[early_mask])
        else:
            near_early_error = 0.0

        all_results['near_ood']['accuracy'].append(near_acc)
        all_results['near_ood']['disagreement_rate'].append(near_disagree)
        all_results['near_ood']['mean_stop_time'].append(near_mean_stop)
        all_results['near_ood']['work_reduction'].append(near_work_reduction)
        all_results['near_ood']['early_exit_error_rate'].append(near_early_error)

        logger.info(f"Near OOD: Acc={near_acc:.4f}, Disagree={near_disagree:.4f}, "
                   f"MeanStop={near_mean_stop:.1f}/{T}, EarlyErr={near_early_error:.4f}")

        # Evaluate on Far OOD
        logger.info("Evaluating on Far OOD...")
        far_preds, far_stops = engine.run(data['X_far_ood'])
        far_full_preds = prefix_evaluator.get_full_predictions(data['X_far_ood'])

        far_acc = np.mean(far_preds == data['y_far_ood'])
        far_disagree = np.mean(far_preds != far_full_preds)
        far_mean_stop = np.mean(far_stops)
        far_work_reduction = 1 - far_mean_stop / T

        early_mask = far_stops < T
        if np.any(early_mask):
            far_early_error = np.mean(far_preds[early_mask] != far_full_preds[early_mask])
        else:
            far_early_error = 0.0

        all_results['far_ood']['accuracy'].append(far_acc)
        all_results['far_ood']['disagreement_rate'].append(far_disagree)
        all_results['far_ood']['mean_stop_time'].append(far_mean_stop)
        all_results['far_ood']['work_reduction'].append(far_work_reduction)
        all_results['far_ood']['early_exit_error_rate'].append(far_early_error)

        logger.info(f"Far OOD: Acc={far_acc:.4f}, Disagree={far_disagree:.4f}, "
                   f"MeanStop={far_mean_stop:.1f}/{T}, EarlyErr={far_early_error:.4f}")

        # OOD detection using suspicion scores
        if suspicion_model is not None:
            # Only compute if model supports static scoring (not dynamic)
            id_suspicion = suspicion_model.compute_ood_score(data['X_id_test'])
            if id_suspicion is not None:
                near_suspicion = suspicion_model.compute_ood_score(data['X_near_ood'])
                far_suspicion = suspicion_model.compute_ood_score(data['X_far_ood'])

                near_ood_metrics = compute_ood_metrics(id_suspicion, near_suspicion)
                far_ood_metrics = compute_ood_metrics(id_suspicion, far_suspicion)

                all_results['ood_detection']['near_auroc'].append(near_ood_metrics['auroc'])
                all_results['ood_detection']['far_auroc'].append(far_ood_metrics['auroc'])

                logger.info(f"OOD Detection: Near AUROC={near_ood_metrics['auroc']:.4f}, "
                           f"Far AUROC={far_ood_metrics['auroc']:.4f}")

        run_time = time.time() - start_time
        all_results['run_times'].append(run_time)
        logger.info(f"Run completed in {run_time:.2f}s")

    # Compute statistics
    for key in ['id_test', 'near_ood', 'far_ood']:
        for metric in list(all_results[key].keys()):
            values = all_results[key][metric]
            if isinstance(values, list) and len(values) > 0:
                all_results[key][f'{metric}_mean'] = float(np.mean(values))
                all_results[key][f'{metric}_std'] = float(np.std(values))

    for metric in list(all_results['calibration'].keys()):
        values = all_results['calibration'][metric]
        if isinstance(values, list) and len(values) > 0:
            all_results['calibration'][f'{metric}_mean'] = float(np.mean(values))
            all_results['calibration'][f'{metric}_std'] = float(np.std(values))

    for metric in list(all_results['ood_detection'].keys()):
        if all_results['ood_detection'][metric]:
            values = all_results['ood_detection'][metric]
            if isinstance(values, list) and len(values) > 0:
                all_results['ood_detection'][f'{metric}_mean'] = float(np.mean(values))
                all_results['ood_detection'][f'{metric}_std'] = float(np.std(values))

    return all_results


def main(iteration=2):
    """Run proposed algorithm experiment."""
    logger = setup_logging(iteration)

    # Iteration-specific configurations
    iteration_configs = {
        1: {'hard_gate_threshold': 0.8},
        2: {'hard_gate_threshold': 0.6},  # Lower threshold to reduce OOD early exit errors
        3: {'hard_gate_threshold': 0.5},  # Even lower if needed
        3: {'hard_gate_threshold': 0.6, 'suspicion_type': 'trajectory'},  # Iteration 3: Trajectory suspicion
        4: {'hard_gate_threshold': 0.7, 'delta_suspicious': 0.005},
    }

    config = iteration_configs.get(iteration, {'hard_gate_threshold': 0.6})
    hard_gate_threshold = config.get('hard_gate_threshold', 0.6)
    delta_suspicious = config.get('delta_suspicious', 0.01)
    suspicion_type = config.get('suspicion_type', 'knn')

    logger.info("="*60)
    logger.info(f"ITERATION {iteration} - Risk-Aware Adaptive Inference")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Config: hard_gate_threshold={hard_gate_threshold}, delta_suspicious={delta_suspicious}, suspicion_type={suspicion_type}")
    logger.info("="*60)

    all_results = {
        'iteration': iteration,
        'timestamp': datetime.now().isoformat()
    }

    # Run for each dataset and model type
    for dataset in ['covertype', 'higgs', 'mnist']:
        for model_type in ['rf', 'gbm']:
            key = f"{dataset}_{model_type}"
            try:
                results = run_proposed_algorithm(
                    dataset, model_type=model_type, n_runs=3, seed=42,
                    delta_id=0.05, delta_suspicious=delta_suspicious,
                    hard_gate_threshold=hard_gate_threshold,
                    use_calibration=True, use_suspicion=True,
                    suspicion_type=suspicion_type
                )
                all_results[key] = results
            except Exception as e:
                logger.error(f"Failed for {key}: {e}")
                import traceback
                traceback.print_exc()

    # Save results
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'iteration_{iteration}.json')

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("PROPOSED ALGORITHM RESULTS SUMMARY")
    logger.info("="*60)

    for key in all_results:
        if key in ['iteration', 'timestamp']:
            continue
        results = all_results[key]
        logger.info(f"\n{key}:")
        logger.info(f"  ID Accuracy: {results['id_test']['accuracy_mean']:.4f} +/- {results['id_test']['accuracy_std']:.4f}")
        logger.info(f"  ID Work Reduction: {results['id_test']['work_reduction_mean']:.2%}")
        logger.info(f"  ID Disagreement: {results['id_test']['disagreement_rate_mean']:.4f}")
        logger.info(f"  Near OOD Early Exit Error: {results['near_ood']['early_exit_error_rate_mean']:.4f}")
        logger.info(f"  Far OOD Early Exit Error: {results['far_ood']['early_exit_error_rate_mean']:.4f}")

    return all_results


if __name__ == '__main__':
    import sys
    iteration = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(iteration)
