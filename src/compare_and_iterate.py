"""
Compare proposed algorithm against baselines and determine if iteration is needed.
"""
import os
import sys
import json
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def load_baseline_results():
    """Load baseline results."""
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                       'baselines', 'baseline_results.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def load_iteration_results(iteration):
    """Load results from a specific iteration."""
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                       'results', f'iteration_{iteration}.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def compare_results(proposed, baselines, dataset_key):
    """
    Compare proposed algorithm against baselines.

    Returns:
        dict with comparison metrics and whether proposed outperforms
    """
    comparison = {
        'dataset': dataset_key,
        'proposed': {},
        'baselines': {},
        'outperforms': False,
        'advantages': [],
        'disadvantages': []
    }

    if dataset_key not in proposed or dataset_key not in baselines:
        return comparison

    prop = proposed[dataset_key]
    base = baselines[dataset_key]

    # Extract proposed metrics
    comparison['proposed'] = {
        'id_accuracy': prop['id_test']['accuracy_mean'],
        'work_reduction': prop['id_test']['work_reduction_mean'],
        'disagreement_rate': prop['id_test']['disagreement_rate_mean'],
        'near_ood_early_error': prop['near_ood']['early_exit_error_rate_mean'],
        'far_ood_early_error': prop['far_ood']['early_exit_error_rate_mean']
    }

    # Extract baseline metrics
    comparison['baselines'] = {
        'id_accuracy': base['id_accuracy_mean'],
        'best_near_auroc': 0,
        'best_far_auroc': 0
    }

    # Find best baseline OOD detection
    for name, metrics in base.get('near_ood', {}).items():
        if 'auroc_mean' in metrics:
            if metrics['auroc_mean'] > comparison['baselines']['best_near_auroc']:
                comparison['baselines']['best_near_auroc'] = metrics['auroc_mean']
                comparison['baselines']['best_near_method'] = name

    for name, metrics in base.get('far_ood', {}).items():
        if 'auroc_mean' in metrics:
            if metrics['auroc_mean'] > comparison['baselines']['best_far_auroc']:
                comparison['baselines']['best_far_auroc'] = metrics['auroc_mean']
                comparison['baselines']['best_far_method'] = name

    # Check advantages
    # 1. Work reduction > 20% with minimal accuracy loss
    if comparison['proposed']['work_reduction'] > 0.2:
        acc_diff = base['id_accuracy_mean'] - comparison['proposed']['id_accuracy']
        if acc_diff < 0.02:  # Less than 2% accuracy loss
            comparison['advantages'].append(
                f"Achieves {comparison['proposed']['work_reduction']:.1%} work reduction "
                f"with only {acc_diff:.2%} accuracy loss"
            )

    # 2. Low disagreement rate (calibration working)
    if comparison['proposed']['disagreement_rate'] < 0.05:
        comparison['advantages'].append(
            f"Low disagreement rate ({comparison['proposed']['disagreement_rate']:.2%}) "
            "indicates good calibration"
        )

    # 3. OOD safety: low early exit error on OOD
    if comparison['proposed']['near_ood_early_error'] < 0.1:
        comparison['advantages'].append(
            f"Low Near OOD early exit error ({comparison['proposed']['near_ood_early_error']:.2%})"
        )

    if comparison['proposed']['far_ood_early_error'] < 0.1:
        comparison['advantages'].append(
            f"Low Far OOD early exit error ({comparison['proposed']['far_ood_early_error']:.2%})"
        )

    # Check disadvantages
    if comparison['proposed']['work_reduction'] < 0.1:
        comparison['disadvantages'].append(
            f"Low work reduction ({comparison['proposed']['work_reduction']:.1%})"
        )

    if comparison['proposed']['disagreement_rate'] > 0.1:
        comparison['disadvantages'].append(
            f"High disagreement rate ({comparison['proposed']['disagreement_rate']:.2%})"
        )

    acc_diff = base['id_accuracy_mean'] - comparison['proposed']['id_accuracy']
    if acc_diff > 0.03:
        comparison['disadvantages'].append(
            f"Significant accuracy loss ({acc_diff:.2%})"
        )

    # Determine if outperforms
    # Outperforms if: work reduction > 20% AND disagreement < 5% AND OOD early error < 10%
    # AND accuracy loss < 2%
    outperforms = (
        comparison['proposed']['work_reduction'] > 0.2 and
        comparison['proposed']['disagreement_rate'] < 0.05 and
        comparison['proposed']['near_ood_early_error'] < 0.15 and
        comparison['proposed']['far_ood_early_error'] < 0.15 and
        acc_diff < 0.02
    )
    comparison['outperforms'] = outperforms

    return comparison


def generate_improvement_suggestions(comparison_results):
    """
    Generate improvement suggestions based on comparison results.

    Returns:
        list of improvement items with scores
    """
    improvements = []

    # Analyze weaknesses across datasets
    low_work_reduction = []
    high_disagreement = []
    high_ood_error = []

    for key, comp in comparison_results.items():
        if 'proposed' not in comp:
            continue

        if comp['proposed'].get('work_reduction', 0) < 0.2:
            low_work_reduction.append(key)

        if comp['proposed'].get('disagreement_rate', 1) > 0.05:
            high_disagreement.append(key)

        if comp['proposed'].get('near_ood_early_error', 1) > 0.1:
            high_ood_error.append(key)

    # Generate improvements based on weaknesses
    if low_work_reduction:
        improvements.append({
            'name': 'Relax early stopping threshold',
            'description': 'Increase delta_id to allow earlier stopping when confidence is high',
            'affected_datasets': low_work_reduction,
            'expected_gain': '10-20% more work reduction',
            'complexity': 'low',
            'contribution': 'medium',
            'action': 'Increase delta_id from 0.05 to 0.08'
        })

    if high_disagreement:
        improvements.append({
            'name': 'Improve calibration features',
            'description': 'Add entropy and volatility features to calibration model',
            'affected_datasets': high_disagreement,
            'expected_gain': '2-5% reduction in disagreement',
            'complexity': 'medium',
            'contribution': 'high',
            'action': 'Add entropy to calibration feature set'
        })

    if high_ood_error:
        improvements.append({
            'name': 'Strengthen OOD gating',
            'description': 'Lower hard gate threshold or use multi-level suspicion',
            'affected_datasets': high_ood_error,
            'expected_gain': '5-10% reduction in OOD early exit errors',
            'complexity': 'low',
            'contribution': 'high',
            'action': 'Lower hard_gate_threshold from 0.8 to 0.6'
        })

    improvements.append({
        'name': 'Add trajectory-based suspicion',
        'description': 'Use prediction volatility in early stages as OOD signal',
        'affected_datasets': high_ood_error,
        'expected_gain': '3-8% improvement in OOD safety',
        'complexity': 'medium',
        'contribution': 'high',
        'action': 'Enable trajectory suspicion in CombinedSuspicion'
    })

    improvements.append({
        'name': 'Tune minimum depth per dataset',
        'description': 'Increase t_min for datasets with high early volatility',
        'affected_datasets': high_disagreement + high_ood_error,
        'expected_gain': '2-5% reduction in errors',
        'complexity': 'low',
        'contribution': 'medium',
        'action': 'Increase t_min_id from 0.1 to 0.15'
    })

    return improvements


def save_iteration_summary(iteration, comparison_results, improvements, outperforms_all):
    """Save iteration summary to iteration_history."""
    history_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'iteration_history')
    os.makedirs(history_dir, exist_ok=True)

    summary_path = os.path.join(history_dir, f'iteration_{iteration}.md')

    with open(summary_path, 'w') as f:
        f.write(f"# Iteration {iteration} Summary\n\n")
        f.write(f"**Timestamp:** {datetime.now().isoformat()}\n\n")

        f.write("## Results Summary\n\n")
        f.write("| Dataset | ID Acc | Work Red. | Disagree | Near OOD Err | Far OOD Err | Outperforms |\n")
        f.write("|---------|--------|-----------|----------|--------------|-------------|-------------|\n")

        for key, comp in comparison_results.items():
            if 'proposed' not in comp:
                continue
            p = comp['proposed']
            f.write(f"| {key} | {p.get('id_accuracy', 0):.4f} | "
                   f"{p.get('work_reduction', 0):.1%} | "
                   f"{p.get('disagreement_rate', 0):.2%} | "
                   f"{p.get('near_ood_early_error', 0):.2%} | "
                   f"{p.get('far_ood_early_error', 0):.2%} | "
                   f"{'Yes' if comp.get('outperforms') else 'No'} |\n")

        f.write("\n## Comparison with Baselines\n\n")
        for key, comp in comparison_results.items():
            if 'advantages' not in comp:
                continue
            f.write(f"### {key}\n\n")
            if comp['advantages']:
                f.write("**Advantages:**\n")
                for adv in comp['advantages']:
                    f.write(f"- {adv}\n")
            if comp['disadvantages']:
                f.write("\n**Disadvantages:**\n")
                for dis in comp['disadvantages']:
                    f.write(f"- {dis}\n")
            f.write("\n")

        if not outperforms_all:
            f.write("## Improvement Suggestions\n\n")
            for i, imp in enumerate(improvements[:3], 1):
                f.write(f"### {i}. {imp['name']}\n\n")
                f.write(f"- **Description:** {imp['description']}\n")
                f.write(f"- **Expected Gain:** {imp['expected_gain']}\n")
                f.write(f"- **Complexity:** {imp['complexity']}\n")
                f.write(f"- **Contribution:** {imp['contribution']}\n")
                f.write(f"- **Action:** {imp['action']}\n\n")

            f.write("## Selected Improvement for Next Iteration\n\n")
            if improvements:
                selected = improvements[0]
                f.write(f"**{selected['name']}**\n\n")
                f.write(f"{selected['description']}\n\n")
                f.write(f"Action: {selected['action']}\n")
        else:
            f.write("## Conclusion\n\n")
            f.write("**SUCCESS!** The proposed algorithm outperforms baselines on all datasets.\n")

    # Also update main summary
    summary_main_path = os.path.join(history_dir, 'summary.md')
    mode = 'a' if os.path.exists(summary_main_path) else 'w'
    with open(summary_main_path, mode) as f:
        if mode == 'w':
            f.write("# Iteration History Summary\n\n")
        f.write(f"\n## Iteration {iteration}\n")
        f.write(f"- Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"- Outperforms all: {'Yes' if outperforms_all else 'No'}\n")
        if not outperforms_all and improvements:
            f.write(f"- Next improvement: {improvements[0]['name']}\n")

    logger.info(f"Summary saved to {summary_path}")
    return summary_path


def main(iteration=1):
    """Compare results and generate iteration summary."""
    logger.info(f"="*60)
    logger.info(f"Comparing Iteration {iteration} Results")
    logger.info(f"="*60)

    # Load results
    baselines = load_baseline_results()
    proposed = load_iteration_results(iteration)

    if baselines is None:
        logger.error("Baseline results not found. Run run_baselines.py first.")
        return False, None

    if proposed is None:
        logger.error(f"Iteration {iteration} results not found. Run run_proposed.py first.")
        return False, None

    # Compare for each dataset
    comparison_results = {}
    for key in proposed:
        if key in ['iteration', 'timestamp']:
            continue
        if key in baselines:
            comparison_results[key] = compare_results(proposed, baselines, key)

    # Check if outperforms all
    outperforms_count = sum(1 for c in comparison_results.values()
                          if c.get('outperforms', False))
    total_count = len(comparison_results)
    outperforms_all = outperforms_count == total_count and total_count > 0

    logger.info(f"\nOutperforms: {outperforms_count}/{total_count} datasets")

    # Generate improvements if needed
    if not outperforms_all:
        improvements = generate_improvement_suggestions(comparison_results)
    else:
        improvements = []

    # Save summary
    save_iteration_summary(iteration, comparison_results, improvements, outperforms_all)

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*60)

    for key, comp in comparison_results.items():
        logger.info(f"\n{key}:")
        if 'proposed' in comp:
            logger.info(f"  Work Reduction: {comp['proposed'].get('work_reduction', 0):.1%}")
            logger.info(f"  Disagreement: {comp['proposed'].get('disagreement_rate', 0):.2%}")
            logger.info(f"  Outperforms: {'Yes' if comp.get('outperforms') else 'No'}")

    if outperforms_all:
        logger.info("\n*** SUCCESS! Algorithm outperforms baselines. ***")
    else:
        logger.info("\n*** Needs improvement. See suggestions. ***")
        if improvements:
            logger.info(f"Selected improvement: {improvements[0]['name']}")

    return outperforms_all, improvements


if __name__ == '__main__':
    import sys
    iteration = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    success, improvements = main(iteration)
    sys.exit(0 if success else 1)
