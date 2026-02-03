"""
Generate tables and figures for the Ablation Study section of the paper.

Outputs:
1. LaTeX table: Pareto-optimal configurations at different safety levels
2. Figures (TIFF, high-resolution) saved under paper/ with Fig*.tiff names
3. Calibration and OOD analysis tables saved under paper_outputs/

Usage:
    python src/generate_ablation_paper_content.py
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MaxNLocator
matplotlib.use('Agg')  # Non-interactive backend

# Set publication-quality defaults
MM_PER_INCH = 25.4
COLUMN_WIDTH_MM = 84
FULL_WIDTH_MM = 174
MAX_HEIGHT_MM = 234
COLUMN_WIDTH_IN = COLUMN_WIDTH_MM / MM_PER_INCH
FULL_WIDTH_IN = FULL_WIDTH_MM / MM_PER_INCH

FIGURE_DIR = 'paper'
FIGURE_DPI = 1200
FIGURE_FORMAT = 'eps'
FIGURE_FILENAMES = {
    'ood_safety': 'Fig1.eps',
    'calibration': 'Fig2.eps',
    'ablation_pareto': 'Fig3.eps',
    'pareto_frontier': 'Fig4.eps',
    'sensitivity': 'Fig5.eps',
}
FIGURE_PNG_DPI = 300

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.figsize': (FULL_WIDTH_IN, 4.0),
    'figure.dpi': FIGURE_DPI,
    'savefig.dpi': FIGURE_DPI,
    'savefig.bbox': 'tight',
    'savefig.transparent': False,
    'axes.grid': True,
    'grid.alpha': 1.0,
    'grid.color': '0.85',
    'grid.linewidth': 0.5,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
})

OUTPUT_DIR = 'paper_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


def get_figure_path(fig_key):
    filename = FIGURE_FILENAMES[fig_key]
    return os.path.join(FIGURE_DIR, filename)


def get_png_path(fig_key):
    eps_name = FIGURE_FILENAMES[fig_key]
    png_name = os.path.splitext(eps_name)[0] + ".png"
    return os.path.join(FIGURE_DIR, png_name)


def save_figure(fig_key):
    plt.savefig(get_figure_path(fig_key), dpi=FIGURE_DPI, format=FIGURE_FORMAT)
    plt.savefig(get_png_path(fig_key), dpi=FIGURE_PNG_DPI, format='png')
    print(f"Saved: {get_figure_path(fig_key)}")
    print(f"Saved: {get_png_path(fig_key)}")


def load_ablation_results(path='results/ablation_full_merged.json'):
    """Load ablation study results."""
    with open(path) as f:
        return json.load(f)


def load_calibration_results(path='results/final/calibration_analysis.json'):
    """Load calibration analysis results."""
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}")
        return {}
    with open(path) as f:
        return json.load(f)


def load_ood_results(proposed_path='results/final/optimal_config_results.json',
                     baselines_path='baselines/comprehensive_results.json'):
    """Load proposed and baseline OOD analysis results."""
    if not os.path.exists(proposed_path):
        print(f"Warning: File not found: {proposed_path}")
        proposed = {}
    else:
        with open(proposed_path) as f:
            proposed = json.load(f)

    if not os.path.exists(baselines_path):
        print(f"Warning: File not found: {baselines_path}")
        baselines = {}
    else:
        with open(baselines_path) as f:
            baselines = json.load(f)

    return proposed, baselines


def mean(lst):
    return sum(lst) / len(lst) if lst else 0


def extract_configs(results):
    """Extract configuration data from results."""
    all_configs = {}
    for combo_name, combo_results in results['results'].items():
        configs = []
        for r in combo_results:
            cfg = r['config']
            max_ood = max(r['near_ood']['disagreement_rate_mean'],
                          r['far_ood']['disagreement_rate_mean'])
            configs.append({
                'gate': cfg['hard_gate_threshold'],
                'delta_id': cfg['delta_id'],
                'delta_sus': cfg['delta_suspicious'],
                'id_work_red': r['id_test']['work_reduction_mean'],
                'id_work_std': r['id_test']['work_reduction_std'],
                'id_disagree': r['id_test']['disagreement_rate_mean'],
                'near_disagree': r['near_ood']['disagreement_rate_mean'],
                'far_disagree': r['far_ood']['disagreement_rate_mean'],
                'max_ood': max_ood,
            })
        all_configs[combo_name] = configs
    return all_configs


def find_pareto_optimal(configs, safety_levels=[0.01, 0.02, 0.03, 0.05]):
    """Find Pareto-optimal configurations at different safety levels."""
    pareto = {}
    for safety in safety_levels:
        safe = [c for c in configs if c['max_ood'] <= safety]
        if safe:
            best = max(safe, key=lambda x: x['id_work_red'])
            pareto[safety] = best
        else:
            pareto[safety] = None
    return pareto


def generate_latex_table(all_configs):
    """Generate LaTeX table of Pareto-optimal configurations."""
    safety_levels = [0.01, 0.02, 0.03, 0.05]

    # Dataset display names
    dataset_names = {
        'covertype_rf': 'Covertype (RF)',
        'covertype_gbm': 'Covertype (GBM)',
        'higgs_rf': 'HIGGS (RF)',
        'higgs_gbm': 'HIGGS (GBM)',
        'mnist_rf': 'MNIST (RF)',
        'mnist_gbm': 'MNIST (GBM)',
    }

    latex = []
    latex.append(r"\begin{table*}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Pareto-optimal configurations at different OOD safety levels. "
                 r"Work reduction (\%) shows compute savings on ID data while maintaining "
                 r"maximum OOD disagreement below the specified threshold. "
                 r"$\tau$: hard gate threshold, $\delta_{id}$: ID risk threshold, "
                 r"$\delta_{sus}$: suspicious sample risk threshold.}")
    latex.append(r"\label{tab:ablation_pareto}")
    latex.append(r"\small")
    latex.append(r"\begin{tabular}{l|ccc|c|ccc|c}")
    latex.append(r"\toprule")
    latex.append(r"& \multicolumn{4}{c|}{$\leq$2\% OOD Disagreement} & \multicolumn{4}{c}{$\leq$5\% OOD Disagreement} \\")
    latex.append(r"Dataset & $\tau$ & $\delta_{id}$ & $\delta_{sus}$ & Work\% & $\tau$ & $\delta_{id}$ & $\delta_{sus}$ & Work\% \\")
    latex.append(r"\midrule")

    for combo_name in ['covertype_rf', 'covertype_gbm', 'higgs_rf', 'higgs_gbm', 'mnist_rf', 'mnist_gbm']:
        configs = all_configs[combo_name]
        pareto = find_pareto_optimal(configs)

        row = [dataset_names[combo_name]]

        # 2% safety level
        if pareto[0.02]:
            p = pareto[0.02]
            row.extend([f"{p['gate']}", f"{p['delta_id']}", f"{p['delta_sus']}",
                       f"{p['id_work_red']*100:.1f}"])
        else:
            row.extend(["--", "--", "--", "--"])

        # 5% safety level
        if pareto[0.05]:
            p = pareto[0.05]
            row.extend([f"{p['gate']}", f"{p['delta_id']}", f"{p['delta_sus']}",
                       f"{p['id_work_red']*100:.1f}"])
        else:
            row.extend(["--", "--", "--", "--"])

        latex.append(" & ".join(row) + r" \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table*}")

    return "\n".join(latex)


def generate_fig4_table(all_configs):
    """Generate LaTeX table summarizing Fig4 operating points."""
    safety_levels = [0.02, 0.03, 0.05]

    dataset_names = {
        'covertype_rf': 'Covertype (RF)',
        'covertype_gbm': 'Covertype (GBM)',
        'higgs_rf': 'HIGGS (RF)',
        'higgs_gbm': 'HIGGS (GBM)',
        'mnist_rf': 'MNIST (RF)',
        'mnist_gbm': 'MNIST (GBM)',
    }

    latex = []
    latex.append(r"\begin{table*}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Fig.\ 4 operating points. Best ID work reduction (WR) under candidate OOD disagreement budgets. Each column reports the configuration with maximum WR subject to max OOD disagreement (max over Near/Far shifts) not exceeding the stated budget.}")
    latex.append(r"\label{tab:fig4_operating_points}")
    latex.append(r"\small")
    latex.append(r"\begin{tabular}{l|ccccc|ccccc|ccccc}")
    latex.append(r"\toprule")
    latex.append(r"& \multicolumn{5}{c|}{$\leq$2\% OOD} & \multicolumn{5}{c|}{$\leq$3\% OOD} & \multicolumn{5}{c}{$\leq$5\% OOD} \\")
    latex.append(r"Dataset & $\tau$ & $\delta_{id}$ & $\delta_{sus}$ & WR\% & Max OOD\% & $\tau$ & $\delta_{id}$ & $\delta_{sus}$ & WR\% & Max OOD\% & $\tau$ & $\delta_{id}$ & $\delta_{sus}$ & WR\% & Max OOD\% \\")
    latex.append(r"\midrule")

    for combo_name in ['covertype_rf', 'covertype_gbm', 'higgs_rf', 'higgs_gbm', 'mnist_rf', 'mnist_gbm']:
        configs = all_configs[combo_name]
        pareto = find_pareto_optimal(configs, safety_levels)

        row = [dataset_names[combo_name]]
        for safety in safety_levels:
            if pareto[safety]:
                p = pareto[safety]
                row.extend([
                    f"{p['gate']}",
                    f"{p['delta_id']}",
                    f"{p['delta_sus']}",
                    f"{p['id_work_red']*100:.1f}",
                    f"{p['max_ood']*100:.1f}",
                ])
            else:
                row.extend(["--", "--", "--", "--", "--"])

        latex.append(" & ".join(row) + r" \\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table*}")

    return "\n".join(latex)


def generate_pareto_figure(all_configs):
    """Generate Pareto frontier figure."""
    fig, axes = plt.subplots(2, 3, figsize=(FULL_WIDTH_IN, 5.2), sharey=True)
    axes = axes.flatten()

    dataset_names = {
        'covertype_rf': 'Covertype (RF)',
        'covertype_gbm': 'Covertype (GBM)',
        'higgs_rf': 'HIGGS (RF)',
        'higgs_gbm': 'HIGGS (GBM)',
        'mnist_rf': 'MNIST (RF)',
        'mnist_gbm': 'MNIST (GBM)',
    }

    combo_order = ['covertype_rf', 'covertype_gbm', 'higgs_rf', 'higgs_gbm', 'mnist_rf', 'mnist_gbm']
    all_max_oods = []
    all_work_reds = []
    for combo_name in combo_order:
        configs = all_configs[combo_name]
        all_max_oods.extend([c['max_ood'] * 100 for c in configs])
        all_work_reds.extend([c['id_work_red'] * 100 for c in configs])

    if all_max_oods:
        global_max_ood = max(all_max_oods)
        if global_max_ood <= 10:
            x_max = 10
        else:
            x_max = max(10, 5 * int(np.ceil(global_max_ood / 5.0)))
    else:
        x_max = 10

    if all_work_reds:
        y_min = min(all_work_reds) * 0.95
        y_max = max(all_work_reds) * 1.02
    else:
        y_min, y_max = 0, 1

    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    for idx, combo_name in enumerate(combo_order):
        ax = axes[idx]
        configs = all_configs[combo_name]

        # Plot all configurations
        work_reds = [c['id_work_red'] * 100 for c in configs]
        max_oods = [c['max_ood'] * 100 for c in configs]

        ax.scatter(max_oods, work_reds, s=14, c='0.75',
                   edgecolors='0.4', linewidths=0.4, label='All configs')

        # Find and plot Pareto frontier
        # Sort by max_ood ascending
        sorted_configs = sorted(configs, key=lambda x: x['max_ood'])

        pareto_frontier = []
        best_work = -1
        for c in sorted_configs:
            if c['id_work_red'] > best_work:
                pareto_frontier.append(c)
                best_work = c['id_work_red']

        pareto_x = [c['max_ood'] * 100 for c in pareto_frontier]
        pareto_y = [c['id_work_red'] * 100 for c in pareto_frontier]

        ax.plot(pareto_x, pareto_y, color='0.1', linewidth=1.2, label='Pareto frontier')
        ax.scatter(pareto_x, pareto_y, s=24, facecolors='white',
                   edgecolors='0.1', linewidths=0.6, zorder=5)

        # Mark safety thresholds
        for thresh in [2, 3, 5]:
            ax.axvline(x=thresh, color='0.2', linestyle='--', linewidth=1.0)

        ax.set_xlabel('')
        if idx % 3 == 0:
            ax.set_ylabel('ID Work Reduction (%)')
        else:
            ax.set_ylabel('')
        ax.text(0.02, 0.98, panel_labels[idx],
                transform=ax.transAxes, ha='left', va='top')
        ax.set_xlim(0, x_max)
        ax.set_ylim(y_min, y_max)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(labelsize=7)

    legend_handles = [
        matplotlib.lines.Line2D(
            [0], [0], marker='o', linestyle='None',
            markerfacecolor='0.75', markeredgecolor='0.4',
            markersize=4, label='All configs'
        ),
        matplotlib.lines.Line2D(
            [0], [0], marker='o', linestyle='-',
            color='0.1', markerfacecolor='white',
            markeredgecolor='0.1', linewidth=1.2,
            markersize=4, label='Pareto frontier'
        ),
        matplotlib.lines.Line2D(
            [0], [0], linestyle='--', color='0.2',
            linewidth=1.0, label='Candidate OOD budgets'
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.08),
        ncol=3,
        frameon=False,
        fontsize=8,
        handlelength=2.0,
        columnspacing=1.2
    )

    fig.supxlabel('Max OOD disagreement with full ensemble (%, max over shifts)', y=0.145)
    panel_map = ", ".join(
        [f"{panel_labels[i]} {dataset_names[name]}" for i, name in enumerate(combo_order)]
    )
    fig.text(0.5, 0.055, panel_map, ha='center', va='bottom', fontsize=7)
    plt.tight_layout(rect=[0.04, 0.11, 1, 0.90])
    fig.subplots_adjust(wspace=0.28, hspace=0.32)
    save_figure('pareto_frontier')
    plt.close()


def get_pareto_frontier(points):
    """
    Find Pareto frontier for maximizing x (work reduction) and minimizing y (disagreement).
    points: list of (x, y) tuples.
    """
    sorted_points = sorted(points, key=lambda p: (p[1], -p[0]))
    pareto = []
    max_x = -float('inf')
    for x, y in sorted_points:
        if x > max_x:
            pareto.append((x, y))
            max_x = x
    return pareto


def generate_ablation_gate_plot(results):
    """Generate ablation pareto plot with hard gate threshold coloring."""
    data = results.get('results', {})
    datasets = ['mnist', 'covertype', 'higgs']
    models = ['rf', 'gbm']

    fig, axes = plt.subplots(2, 3, figsize=(FULL_WIDTH_IN, 4.8), sharey=True)

    gate_values = [0.5, 0.6, 0.7, 0.8, 0.9]
    gate_markers = ['o', 's', '^', 'D', 'P']
    gate_colors = ['0.2', '0.35', '0.5', '0.65', '0.8']
    gate_style = dict(zip(gate_values, zip(gate_markers, gate_colors)))

    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            ax = axes[j, i]
            key = f"{dataset}_{model}"

            if key not in data:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                continue

            runs = data[key]
            points = []  # (work_reduction, disagreement, gate_val)

            for run in runs:
                cfg = run['config']
                metrics_id = run['id_test']
                metrics_ood = run['far_ood']

                x = metrics_id.get('work_reduction_mean', 0)
                y = metrics_ood.get('disagreement_rate_mean', 0)
                gate = cfg.get('hard_gate_threshold', 0.5)

                points.append((x, y, gate))

            if not points:
                ax.text(0.5, 0.5, "No Points", ha='center', va='center')
                continue

            for gate in gate_values:
                gate_points = [(x, y) for x, y, g in points if g == gate]
                if not gate_points:
                    continue
                xs = [p[0] for p in gate_points]
                ys = [p[1] for p in gate_points]
                marker, color = gate_style[gate]
                ax.scatter(xs, ys, marker=marker, c=color,
                           edgecolors='k', linewidths=0.5, s=35,
                           label=f"$\\tau_{{gate}}={gate}$")

            xy_points = [(p[0], p[1]) for p in points]
            pareto = get_pareto_frontier(xy_points)
            pareto.sort(key=lambda p: p[0])
            px = [p[0] for p in pareto]
            py = [p[1] for p in pareto]

            ax.plot(px, py, color='0.1', linestyle='--', linewidth=1.2,
                    label='Pareto Frontier')

            if j == 1:
                ax.set_xlabel("ID Work Reduction")
            else:
                ax.set_xlabel("")
            ax.set_ylabel("")
            ax.grid(True, linestyle='--')

            ax.axhline(y=0.05, color='0.2', linestyle=':', linewidth=1.0,
                       label='5% Disagreement')

            panel_idx = i + 3 * j
            ax.text(0.02, 0.98, panel_labels[panel_idx],
                    transform=ax.transAxes, ha='left', va='top')

    legend_handles = [
        matplotlib.lines.Line2D(
            [0], [0], marker=gate_style[g][0], linestyle='None',
            markerfacecolor=gate_style[g][1], markeredgecolor='k',
            markersize=5, label=f"$\\tau_{{gate}}={g}$"
        ) for g in gate_values
    ]
    legend_handles.append(
        matplotlib.lines.Line2D(
            [0], [0], color='0.1', linestyle='--', linewidth=1.2,
            label='Pareto Frontier'
        )
    )
    legend_handles.append(
        matplotlib.lines.Line2D(
            [0], [0], color='0.2', linestyle=':', linewidth=1.0,
            label='5% Disagreement'
        )
    )
    fig.legend(handles=legend_handles, loc='lower center',
               bbox_to_anchor=(0.5, 0.075), ncol=4,
               frameon=False, fontsize=8)

    os.makedirs(FIGURE_DIR, exist_ok=True)
    panel_names = {'mnist': 'MNIST', 'covertype': 'Covertype', 'higgs': 'HIGGS'}
    panel_map_entries = []
    for j, model in enumerate(models):
        for i, dataset in enumerate(datasets):
            panel_idx = i + 3 * j
            panel_map_entries.append(
                f"{panel_labels[panel_idx]} {panel_names[dataset]} ({model.upper()})"
            )
    panel_map = ", ".join(panel_map_entries)
    fig.text(0.5, 0.055, panel_map, ha='center', va='bottom', fontsize=7)
    fig.text(0.06, 0.55, "Far OOD Disagreement Rate", rotation=90,
             ha='center', va='center', fontsize=plt.rcParams['axes.labelsize'])
    plt.tight_layout(rect=[0.07, 0.18, 1, 0.92])
    fig.subplots_adjust(wspace=0.28, hspace=0.32)
    save_figure('ablation_pareto')
    plt.close()


def aggregate_reliability_data(data):
    """
    Aggregate per-depth reliability data into global reliability curve.
    """
    bins = np.array(data['bins'])
    accuracy = np.array(data['accuracy'])
    confidence = np.array(data['confidence'])
    count = np.array(data['count'])

    unique_bins = np.unique(bins)
    unique_bins.sort()

    agg_accuracy = []
    agg_confidence = []
    agg_count = []

    for b in unique_bins:
        mask = (bins == b)
        total_count = np.sum(count[mask])
        if total_count > 0:
            weighted_acc = np.sum(accuracy[mask] * count[mask]) / total_count
            weighted_conf = np.sum(confidence[mask] * count[mask]) / total_count

            agg_accuracy.append(weighted_acc)
            agg_confidence.append(weighted_conf)
            agg_count.append(total_count)

    return np.array(agg_confidence), np.array(agg_accuracy), np.array(agg_count)


def generate_calibration_plots_and_table(calibration_results):
    """Generate calibration plots and table from calibration analysis results."""
    data = calibration_results.get('results', {})

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 3.2), sharey=True)

    plot_models = [
        ('covertype_rf', 'Covertype RF'),
        ('mnist_gbm', 'MNIST GBM')
    ]

    for i, (key, title) in enumerate(plot_models):
        ax = axes[i]
        if key in data:
            rel_data = data[key]['reliability_data']
            conf, acc, _ = aggregate_reliability_data(rel_data)

            ax.plot([0, 1], [0, 1], color='0.2', linestyle='--',
                    label='Perfect Calibration', linewidth=1.0)

            sort_idx = np.argsort(conf)
            ax.plot(conf[sort_idx], acc[sort_idx], marker='o', linestyle='-',
                    color='0.1', label=title, linewidth=1.2, markersize=4)

            ax.set_xlabel('Predicted Probability of Agreement')
            if i == 0:
                ax.set_ylabel('Empirical Agreement')
            else:
                ax.set_ylabel('')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True)
            ax.text(0.02, 0.98, f"({chr(ord('a') + i)})",
                    transform=ax.transAxes, ha='left', va='top')
            ax.legend(frameon=False)
        else:
            ax.text(0.5, 0.5, f"Data for {title} not found",
                    ha='center', va='center')

    plt.tight_layout()
    save_figure('calibration')
    plt.close()

    table_path = os.path.join(OUTPUT_DIR, 'table_calibration.tex')
    with open(table_path, 'w') as f:
        print(r"\begin{table}[t]", file=f)
        print(r"\centering", file=f)
        print(r"\caption{Calibration Quality. Expected Calibration Error (ECE) and Maximum Calibration Error (MCE) for the flip-risk estimator.}", file=f)
        print(r"\label{tab:calibration}", file=f)
        print(r"\begin{tabular}{llcc}", file=f)
        print(r"\toprule", file=f)
        print(r"\textbf{Dataset} & \textbf{Model} & \textbf{ECE} & \textbf{MCE} \\", file=f)
        print(r"\midrule", file=f)

        dataset_map = {
            'covertype': 'Covertype',
            'higgs': 'HIGGS',
            'mnist': 'MNIST'
        }

        sorted_keys = sorted(data.keys())
        for key in sorted_keys:
            parts = key.split('_')
            if len(parts) < 2:
                continue
            dataset_key = parts[0]
            model_key = parts[1]

            dataset_name = dataset_map.get(dataset_key, dataset_key.capitalize())
            model_name = model_key.upper()

            metrics = data[key]
            ece = metrics.get('ece', 0.0)
            mce = metrics.get('mce', 0.0)

            print(f"{dataset_name} & {model_name} & {ece:.4f} & {mce:.4f} \\\\", file=f)

        print(r"\bottomrule", file=f)
        print(r"\end{tabular}", file=f)
        print(r"\end{table}", file=f)

    print(f"Saved: {table_path}")


def generate_ood_analysis_outputs(proposed_results, baseline_results):
    """Generate OOD analysis figure and detection table."""
    prop_data = proposed_results.get('results', {})
    base_data = baseline_results.get('results', {})

    datasets = ['mnist', 'covertype', 'higgs']
    models = ['rf', 'gbm']
    splits = ['near_ood', 'far_ood']
    methods = ['Proposed', 'Uncalibrated Lazy', 'Constant Threshold']

    plot_data = {split: {'values': {m: [] for m in methods}} for split in splits}
    dataset_labels = []

    for dataset in datasets:
        for model in models:
            key = f"{dataset}_{model}"
            label = f"{dataset.upper()} {model.upper()}"
            dataset_labels.append(label)

            if key not in prop_data or key not in base_data:
                for split in splits:
                    for m in methods:
                        plot_data[split]['values'][m].append(0)
                continue

            p_agg = prop_data[key]['aggregated']
            b_agg = base_data[key]['aggregated']['early_exit']

            for split in splits:
                val_p = p_agg[split].get('early_exit_error_rate_mean', 0)
                plot_data[split]['values']['Proposed'].append(val_p)

                val_u = b_agg['uncalibrated_lazy'][split].get('early_exit_error_rate_mean', 0)
                plot_data[split]['values']['Uncalibrated Lazy'].append(val_u)

                val_c = b_agg['constant_threshold'][split].get('early_exit_error_rate_mean', 0)
                plot_data[split]['values']['Constant Threshold'].append(val_c)

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 3.4), sharey=True)
    x = np.arange(len(dataset_labels))
    width = 0.25
    bar_styles = {
        'Proposed': {'color': '0.2', 'hatch': '///'},
        'Constant Threshold': {'color': '0.5', 'hatch': '\\\\\\'},
        'Uncalibrated Lazy': {'color': '0.8', 'hatch': 'xx'},
    }

    for i, split in enumerate(splits):
        ax = axes[i]
        split_name = "Near OOD" if split == 'near_ood' else "Far OOD"
        vals = plot_data[split]['values']

        ax.bar(x - width, vals['Proposed'], width,
               label='Proposed', edgecolor='k', linewidth=0.6,
               color=bar_styles['Proposed']['color'],
               hatch=bar_styles['Proposed']['hatch'])
        ax.bar(x, vals['Constant Threshold'], width,
               label='Constant Threshold', edgecolor='k', linewidth=0.6,
               color=bar_styles['Constant Threshold']['color'],
               hatch=bar_styles['Constant Threshold']['hatch'])
        ax.bar(x + width, vals['Uncalibrated Lazy'], width,
               label='Uncalibrated Lazy', edgecolor='k', linewidth=0.6,
               color=bar_styles['Uncalibrated Lazy']['color'],
               hatch=bar_styles['Uncalibrated Lazy']['hatch'])

        if i == 0:
            ax.set_ylabel('Early Exit Error Rate')
        else:
            ax.set_ylabel('')
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_labels, rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--')
        ax.text(0.02, 0.98, f"({chr(ord('a') + i)})",
                transform=ax.transAxes, ha='left', va='top')

        if i == 0:
            ax.legend(frameon=False, fontsize=8, ncol=1)

    plt.tight_layout()
    save_figure('ood_safety')
    plt.close()

    table_path = os.path.join(OUTPUT_DIR, 'table_ood_detection.tex')
    with open(table_path, 'w') as f:
        print(r"\begin{table*}[t]", file=f)
        print(r"\centering", file=f)
        print(r"\caption{OOD Detection Performance (AUROC). Comparison of the proposed KNN-based suspicion score against standard OOD detection baselines on Near and Far shifts.}", file=f)
        print(r"\label{tab:ood_detection}", file=f)
        print(r"\resizebox{\textwidth}{!}{", file=f)
        print(r"\begin{tabular}{ll|c|ccc}", file=f)
        print(r"\toprule", file=f)
        print(r"Dataset & Shift & \textbf{Proposed (KNN)} & \textbf{Mahalanobis} & \textbf{Entropy} & \textbf{MSP} \\", file=f)
        print(r"\midrule", file=f)

        for dataset in datasets:
            for model in models:
                key = f"{dataset}_{model}"
                if key not in prop_data or key not in base_data:
                    continue

                p_agg = prop_data[key]['aggregated']
                b_ood = base_data[key]['aggregated']['ood_detection']

                row_near = [f"{dataset.upper()} {model.upper()}", "Near"]
                prop_near = p_agg.get('near_auroc_mean', 0)
                row_near.append(f"{prop_near:.3f}")
                for method in ['mahalanobis', 'entropy', 'msp']:
                    val = b_ood['near_ood'][method].get('auroc_mean', 0)
                    row_near.append(f"{val:.3f}")
                print(" & ".join(row_near) + r" \\", file=f)

                row_far = [f"{dataset.upper()} {model.upper()}", "Far"]
                prop_far = p_agg.get('far_auroc_mean', 0)
                row_far.append(f"{prop_far:.3f}")
                for method in ['mahalanobis', 'entropy', 'msp']:
                    val = b_ood['far_ood'][method].get('auroc_mean', 0)
                    row_far.append(f"{val:.3f}")

                print(" & ".join(row_far) + r" \\", file=f)
                print(r"\midrule", file=f)

        print(r"\bottomrule", file=f)
        print(r"\end{tabular}", file=f)
        print(r"}", file=f)
        print(r"\end{table*}", file=f)

    print(f"Saved: {table_path}")


def generate_sensitivity_figure(all_configs, results):
    """Generate hyperparameter sensitivity figure."""
    fig, axes = plt.subplots(2, 3, figsize=(FULL_WIDTH_IN, 5.2))

    dataset_markers = {
        'covertype': 'o',
        'higgs': 's',
        'mnist': '^',
    }
    dataset_colors = {
        'covertype': '0.2',
        'higgs': '0.45',
        'mnist': '0.7',
    }
    model_linestyles = {
        'rf': '-',
        'gbm': '--',
    }
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    # Row 1: Effect on ID Work Reduction
    # Col 1: hard_gate_threshold
    ax = axes[0, 0]
    for combo_name, configs in all_configs.items():
        dataset, model = combo_name.split('_', 1)
        gate_effect = {}
        for c in configs:
            g = c['gate']
            if g not in gate_effect:
                gate_effect[g] = []
            gate_effect[g].append(c['id_work_red'] * 100)

        gates = sorted(gate_effect.keys())
        means = [mean(gate_effect[g]) for g in gates]
        ax.plot(gates, means, color=dataset_colors[dataset],
                linestyle=model_linestyles[model],
                marker=dataset_markers[dataset], markersize=4,
                label=combo_name.replace('_', ' ').title())

    ax.set_xlabel(r'Hard Gate Threshold ($\tau$)')
    ax.set_ylabel('ID Work Reduction (%)')
    ax.text(0.02, 0.98, panel_labels[0],
            transform=ax.transAxes, ha='left', va='top')
    ax.legend(fontsize=7, ncol=2, loc='lower right', frameon=False)

    # Col 2: delta_id
    ax = axes[0, 1]
    for combo_name, configs in all_configs.items():
        dataset, model = combo_name.split('_', 1)
        delta_effect = {}
        for c in configs:
            d = c['delta_id']
            if d not in delta_effect:
                delta_effect[d] = []
            delta_effect[d].append(c['id_work_red'] * 100)

        deltas = sorted(delta_effect.keys())
        means = [mean(delta_effect[d]) for d in deltas]
        ax.plot(deltas, means, color=dataset_colors[dataset],
                linestyle=model_linestyles[model],
                marker=dataset_markers[dataset], markersize=4)

    ax.set_xlabel(r'ID Risk Threshold ($\delta_{id}$)')
    ax.set_ylabel('ID Work Reduction (%)')
    ax.text(0.02, 0.98, panel_labels[1],
            transform=ax.transAxes, ha='left', va='top')

    # Col 3: delta_suspicious (on work reduction)
    ax = axes[0, 2]
    for combo_name, configs in all_configs.items():
        dataset, model = combo_name.split('_', 1)
        delta_effect = {}
        for c in configs:
            d = c['delta_sus']
            if d not in delta_effect:
                delta_effect[d] = []
            delta_effect[d].append(c['id_work_red'] * 100)

        deltas = sorted(delta_effect.keys())
        means = [mean(delta_effect[d]) for d in deltas]
        ax.plot(deltas, means, color=dataset_colors[dataset],
                linestyle=model_linestyles[model],
                marker=dataset_markers[dataset], markersize=4)

    ax.set_xlabel(r'Suspicious Risk Threshold ($\delta_{sus}$)')
    ax.set_ylabel('ID Work Reduction (%)')
    ax.text(0.02, 0.98, panel_labels[2],
            transform=ax.transAxes, ha='left', va='top')
    ax.set_xscale('log')

    # Row 2: Effect on OOD Safety
    # Col 1: hard_gate_threshold on Near OOD
    ax = axes[1, 0]
    for combo_name, configs in all_configs.items():
        dataset, model = combo_name.split('_', 1)
        gate_effect = {}
        for c in configs:
            g = c['gate']
            if g not in gate_effect:
                gate_effect[g] = []
            gate_effect[g].append(c['near_disagree'] * 100)

        gates = sorted(gate_effect.keys())
        means = [mean(gate_effect[g]) for g in gates]
        ax.plot(gates, means, color=dataset_colors[dataset],
                linestyle=model_linestyles[model],
                marker=dataset_markers[dataset], markersize=4)

    ax.set_xlabel(r'Hard Gate Threshold ($\tau$)')
    ax.set_ylabel('Near OOD Disagreement (%)')
    ax.text(0.02, 0.98, panel_labels[3],
            transform=ax.transAxes, ha='left', va='top')
    ax.axhline(y=5, color='0.2', linestyle='--', linewidth=1.0, label='5% threshold')

    # Col 2: delta_id on ID disagreement
    ax = axes[1, 1]
    for combo_name, configs in all_configs.items():
        dataset, model = combo_name.split('_', 1)
        delta_effect = {}
        for c in configs:
            d = c['delta_id']
            if d not in delta_effect:
                delta_effect[d] = []
            delta_effect[d].append(c['id_disagree'] * 100)

        deltas = sorted(delta_effect.keys())
        means = [mean(delta_effect[d]) for d in deltas]
        ax.plot(deltas, means, color=dataset_colors[dataset],
                linestyle=model_linestyles[model],
                marker=dataset_markers[dataset], markersize=4)

    ax.set_xlabel(r'ID Risk Threshold ($\delta_{id}$)')
    ax.set_ylabel('ID Disagreement (%)')
    ax.text(0.02, 0.98, panel_labels[4],
            transform=ax.transAxes, ha='left', va='top')

    # Col 3: delta_suspicious on Near OOD disagreement
    ax = axes[1, 2]
    for combo_name, configs in all_configs.items():
        dataset, model = combo_name.split('_', 1)
        delta_effect = {}
        for c in configs:
            d = c['delta_sus']
            if d not in delta_effect:
                delta_effect[d] = []
            delta_effect[d].append(c['near_disagree'] * 100)

        deltas = sorted(delta_effect.keys())
        means = [mean(delta_effect[d]) for d in deltas]
        ax.plot(deltas, means, color=dataset_colors[dataset],
                linestyle=model_linestyles[model],
                marker=dataset_markers[dataset], markersize=4)

    ax.set_xlabel(r'Suspicious Risk Threshold ($\delta_{sus}$)')
    ax.set_ylabel('Near OOD Disagreement (%)')
    ax.text(0.02, 0.98, panel_labels[5],
            transform=ax.transAxes, ha='left', va='top')
    ax.set_xscale('log')
    ax.axhline(y=5, color='0.2', linestyle='--', linewidth=1.0)

    plt.tight_layout()
    save_figure('sensitivity')
    plt.close()


def main():
    print("Loading ablation results...")
    results = load_ablation_results()

    print("Extracting configurations...")
    all_configs = extract_configs(results)

    print("\nGenerating LaTeX table...")
    latex_table = generate_latex_table(all_configs)

    table_path = os.path.join(OUTPUT_DIR, 'table_ablation_pareto.tex')
    with open(table_path, 'w') as f:
        f.write(latex_table)
    print(f"Saved: {table_path}")

    print("\nGenerating Fig4 operating-points table...")
    fig4_table = generate_fig4_table(all_configs)
    fig4_table_path = os.path.join(OUTPUT_DIR, 'table_fig4_operating_points.tex')
    with open(fig4_table_path, 'w') as f:
        f.write(fig4_table)
    print(f"Saved: {fig4_table_path}")

    print("\nGenerating Pareto frontier figure...")
    generate_pareto_figure(all_configs)

    print("\nGenerating ablation gate-colored Pareto figure...")
    generate_ablation_gate_plot(results)

    print("\nGenerating calibration plots and table...")
    calibration_results = load_calibration_results()
    generate_calibration_plots_and_table(calibration_results)

    print("\nGenerating OOD analysis figure and table...")
    proposed_results, baseline_results = load_ood_results()
    generate_ood_analysis_outputs(proposed_results, baseline_results)

    print("\nGenerating sensitivity figure...")
    generate_sensitivity_figure(all_configs, results)

    print("SUMMARY")
    print("=" * 70)
    print("Graphics program: Matplotlib (Python)")
    print(f"\nGenerated tables in {OUTPUT_DIR}/:")
    print("  1. table_ablation_pareto.tex  - LaTeX table")
    print("  2. table_fig4_operating_points.tex - Fig4 operating points table")
    print("  3. table_calibration.tex      - Calibration metrics table")
    print("  4. table_ood_detection.tex    - OOD detection table")
    print(f"\nGenerated figures in {FIGURE_DIR}/:")
    print(f"  1. {FIGURE_FILENAMES['ood_safety']}        - OOD safety figure")
    print(f"  2. {FIGURE_FILENAMES['calibration']}       - Calibration reliability figure")
    print(f"  3. {FIGURE_FILENAMES['ablation_pareto']}   - Gate-colored Pareto figure")
    print(f"  4. {FIGURE_FILENAMES['pareto_frontier']}   - Pareto frontier figure")
    print(f"  5. {FIGURE_FILENAMES['sensitivity']}       - Sensitivity analysis figure")
    print("  6. Fig1.png - Fig5.png (PNG previews)")

    print("\n" + "=" * 70)
    print("LATEX TABLE PREVIEW")
    print("=" * 70)
    print(latex_table)


if __name__ == '__main__':
    main()
