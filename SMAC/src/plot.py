import os
import json
import numpy as np
from collections import defaultdict
import argparse
from matplotlib.ticker import MaxNLocator

def _collect_alg_runs(results_root, base_filters=None):
    all_data = defaultdict(list)
    if not os.path.exists(results_root):
        return all_data

    for folder in os.listdir(results_root):
        folder_path = os.path.join(results_root, folder)
        if not os.path.isdir(folder_path):
            continue

        if base_filters:
            if not any(b.lower() in folder.lower() for b in base_filters):
                continue

        info_json_path = os.path.join(folder_path, 'info.json')
        config_json_path = os.path.join(folder_path, 'config.json')

        if not (os.path.exists(info_json_path) and os.path.exists(config_json_path)):
            continue

        try:
            with open(config_json_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            alg_name = config.get('name', folder)

            with open(info_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            series = data.get('battle_won_mean')
            if series is None:
                continue
            all_data[alg_name].append(list(series))
        except Exception as e:
            print(f"Parse {folder} error: {e}")
    return all_data

def _resolve_map_results_root(base_dir, map_value):
    candidate = map_value if os.path.isabs(map_value) else os.path.join(base_dir, map_value)
    if not os.path.isdir(candidate): return None, None
    sacred_dir = os.path.join(candidate, 'sacred')
    root = sacred_dir if os.path.isdir(sacred_dir) else candidate
    label = os.path.basename(candidate.rstrip(os.sep))
    return label, root

def plot_marl_results_across_maps(map_results_roots, base_filters=None, out_path="result.pdf", show=False):
    import matplotlib.pyplot as plt
    
    plt.rcParams.update({
        "font.family": "serif",
        "pdf.use14corefonts": True,
        "ps.useafm": True,
        "axes.labelsize": 14, 
        "axes.titlesize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })

    per_map_data = {}
    all_algs = set()
    for map_label, results_root in map_results_roots.items():
        all_data = _collect_alg_runs(results_root, base_filters=base_filters)
        if all_data:
            per_map_data[map_label] = all_data
            all_algs.update(all_data.keys())

    if not per_map_data:
        print("Valid data not found. Please check --base parameter and folder names.")
        return

    alg_names = sorted(all_algs)
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in range(0, 20, 2)]
    alg_color = {name: colors[i % len(colors)] for i, name in enumerate(alg_names)}

    map_labels = sorted(per_map_data.keys(), reverse=True)
    n_maps = len(map_labels)
    cm_to_inch = 1 / 2.54
    total_width_cm = 9.0
    total_height_cm = 7.5

    fig, axes = plt.subplots(1, n_maps, figsize=(total_width_cm * n_maps * cm_to_inch, total_height_cm * cm_to_inch), sharey=False)
    if n_maps == 1: axes = [axes]

    name_map = {
        "qmix": "QMIX",
        "qmix_cadp": "QMIX-CADP",
        "qmix_ptde": "QMIX-PTDE",
        "qmix_decomm": "QMIX-DeComm",
        "qplex": "QPLEX",
        "qplex_cadp": "QPLEX-CADP",
        "qplex_ptde": "QPLEX-PTDE",
        "qplex_decomm": "QPLEX-DeComm"
    }

    for ax, map_label in zip(axes, map_labels):
        data = per_map_data[map_label]
        for alg_name in alg_names:
            if alg_name not in data: continue
            display_name = name_map.get(alg_name.lower(), alg_name.upper())
            y = data[alg_name]
            max_len = max(len(row) for row in y)
            y_padded = [row + [row[-1]] * (max_len - len(row)) for row in y]
            y_mean = np.mean(y_padded, axis=0)
            y_std = np.std(y_padded, axis=0)
            x = np.arange(len(y_mean))

            ax.plot(x, y_mean, label=display_name, color=alg_color[alg_name], linewidth=1.5)
            ax.fill_between(x, y_mean - y_std, y_mean + y_std, color=alg_color[alg_name], alpha=0.2, edgecolor="none")

        ax.set_title(map_label)
        ax.set_xlabel('Steps(×10^6)')
        ax.set_ylabel('Mean Win Rate')
        
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(True, linestyle='--', alpha=0.5)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles, 
        labels, 
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.1),
        ncol=len(labels), 
        columnspacing=1,
        handletextpad=0.4,
        frameon=True,
        edgecolor='black',
        facecolor='white',
        framealpha=1.0,
        fancybox=False
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.80)
    
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    if show: plt.show()
    else: plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='.', help='root path')
    parser.add_argument('--base', nargs='+', choices=['qmix', 'qplex'], help='filter prefix')
    parser.add_argument('--show', action='store_true', help='show plot window')
    args = parser.parse_args()

    map_results_roots = {}
    for name in sorted(os.listdir(args.path)):
        if os.path.isdir(os.path.join(args.path, name)) and name != 'sacred':
            label, root = _resolve_map_results_root(args.path, name)
            if label and root:
                map_results_roots[label] = root

    base_tag = "all" if not args.base else "_".join(args.base)
    out_path = os.path.join(args.path, f"smac_{base_tag}_training_results.pdf")
    plot_marl_results_across_maps(map_results_roots, base_filters=args.base, out_path=out_path, show=args.show)
