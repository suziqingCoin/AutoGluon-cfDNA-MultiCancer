"""Original manuscript Figure S4 plotting functions for Liver_cancer.

Use --list to inspect input requirements and --only to replay selected groups.
Patient-dependent functions read frozen authorized predictions, never fit models.
Public simplified redraws remain available through code/plot_public.py.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'feature_screen'))
from original_plots import ReplayContext, main

def plot_feature_count_curve(context: ReplayContext):
    """Draw AUC_vs_FeatureCount_ReliefF_BarOnlyVal. Uses the included feature/K aggregate inputs only."""
    PLOT_CURVE = context.curve
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    auc_df = pd.read_csv(PLOT_CURVE)
    plt.figure(figsize=(9, 7))
    x = auc_df['Num_Features']
    bar_colors = ['#bd62a8' if val == 19 else '#eb9192' for val in auc_df['Num_Features']]
    plt.bar(x, auc_df['Validation_AUC'], width=1.0, color=bar_colors, edgecolor='black', linewidth=2, label='Validation AUC')
    xticks = sorted(set(plt.xticks()[0].tolist() + [19]))
    plt.xticks(xticks)
    ax = plt.gca()
    for label in ax.get_xticklabels():
        text = label.get_text()
        (x, y) = label.get_position()
        if text == '10':
            label.set_horizontalalignment('right')
            label.set_bbox({'facecolor': 'white', 'alpha': 1, 'edgecolor': 'none'})
        elif text == '19':
            label.set_horizontalalignment('left')
            label.set_bbox({'facecolor': 'white', 'alpha': 1, 'edgecolor': 'none'})
    plt.xlabel('Number of Features', fontsize=19, fontweight='bold')
    plt.ylabel('Validation AUC Score', fontsize=19, fontweight='bold')
    plt.title('Liver cancer–Feature selection', fontsize=24, fontweight='bold')
    plt.yticks(np.arange(0.7, 1.01, 0.05))
    ax.set_ylim(0.7, 1.01)
    plt.grid(False)
    ax.set_xlim(left=0, right=max(auc_df['Num_Features']) + 2)
    ax.xaxis.set_ticks_position('bottom')
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.tick_params(axis='both', width=3, length=8)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontsize(15)
        tick_label.set_fontweight('bold')
    plt.tight_layout()
    plt.savefig('AUC_vs_FeatureCount_ReliefF_BarOnlyVal.png', dpi=300)
    plt.show()

def plot_model_metric_polar(context: ReplayContext, dataset='test'):
    """Draw the original test/train metric polar view; preserve the distinct 35/45 title pads."""
    if dataset not in ('test', 'train'):
        raise ValueError('dataset must be test or train')
    dataset_label = 'Test' if dataset == 'test' else 'Train'
    evaluation_data = context.test_data if dataset == 'test' else context.train_data
    data = context.data
    model_dir = str(context.output_dir)
    TabularPredictor = context.predictor_api
    train_test_split = context.locked_split
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score, recall_score, precision_score
    from matplotlib.cm import get_cmap

    def compute_metrics(y_true, y_pred, y_prob):
        return {'AUPRC': average_precision_score(y_true, y_prob), 'AUROC': roc_auc_score(y_true, y_prob), 'ACC': accuracy_score(y_true, y_pred), 'F1': f1_score(y_true, y_pred), 'Recall': recall_score(y_true, y_pred), 'Precision': precision_score(y_true, y_pred)}

    def plot_polar_bars(data, num_metrics, num_models):
        (fig, ax) = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        theta = np.linspace(0.0, 2 * np.pi, num_metrics * num_models, endpoint=False)
        radii = data['Score'].values
        width = 2 * np.pi / (num_metrics * num_models)
        colors = {'ACC': '#FEF057', 'Precision': '#FABC6B', 'Recall': '#EB9192', 'F1': '#BD62A8', 'AUROC': '#874AAA', 'AUPRC': '#4F4DA5'}
        ax.set_ylim(0.0, 1.0)
        for r in np.arange(0.0, 1.01, 0.2):
            ax.plot(np.linspace(0, 2 * np.pi, 500), [r] * 500, linestyle='--', color='lightgray', linewidth=1)
            ax.text(np.pi / 2, r, f'{r:.2f}', color='gray', fontsize=8, ha='center', va='bottom', fontweight='bold')
        bars = ax.bar(theta, radii, width=width, bottom=0.0, color=[colors[m] for m in data['Metric']], edgecolor='lightgray', linewidth=1)
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.spines['polar'].set_visible(False)
        for (i, (angle, row)) in enumerate(zip(theta, data.itertuples())):
            rotation = np.degrees(angle)
            if 90 < rotation < 270:
                rotation += 180
                ha = 'right'
            else:
                ha = 'left'
            ax.text(angle, row.Score + 0.005, row.Model, ha=ha, va='center', fontsize=14, fontweight='bold', rotation=rotation, rotation_mode='anchor')
        handles = [plt.Line2D([0], [0], color=c, lw=8) for c in colors.values()]
        labels = list(colors.keys())
        ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.05, 1.1), frameon=False, title='Metrics', title_fontproperties={'weight': 'bold', 'size': 15}, prop={'weight': 'bold', 'size': 14})
        plt.title(f'{dataset_label} - Performance Metric Polar View', fontsize=19, fontweight='bold', pad=35 if dataset == 'test' else 45)
        plt.tight_layout()
        plt.savefig(f'Model_Metric_Polar_Plot_{dataset_label}.png', dpi=300)
        plt.show()
    predictor = TabularPredictor.load(model_dir)
    raw_models = ['XGBoost', 'RandomForest', 'LightGBM', 'NeuralNetFastAI', 'WeightedEnsemble_L2']
    model_name_map = {'XGBoost': 'XGB', 'RandomForest': 'RF', 'LightGBM': 'LGB', 'NeuralNetFastAI': 'NN-FastAI', 'WeightedEnsemble_L2': 'WE'}
    (train_data, test_data) = train_test_split(data, test_size=0.2, stratify=data['class'], random_state=42)
    results = []
    for model in raw_models:
        try:
            y_true = evaluation_data['class']
            y_pred = predictor.predict(evaluation_data, model=model)
            y_prob_df = predictor.predict_proba(evaluation_data, model=model)
            y_prob = y_prob_df.iloc[:, 1] if isinstance(y_prob_df, pd.DataFrame) else y_prob_df
            scores = compute_metrics(y_true, y_pred, y_prob)
            for (metric, score) in scores.items():
                results.append({'Model': model_name_map[model], 'Metric': metric, 'Score': score})
        except Exception as e:
            print(f'Model {model} failed: {e}')
    results_df = pd.DataFrame(results)
    metric_order = ['ACC', 'Precision', 'Recall', 'F1', 'AUROC', 'AUPRC']
    results_df['Metric'] = pd.Categorical(results_df['Metric'], categories=metric_order, ordered=True)
    results_df.sort_values(['Metric', 'Model'], inplace=True)
    plot_polar_bars(results_df, num_metrics=6, num_models=5)

def plot_top_feature_scores(context: ReplayContext):
    """Draw Top_Feature_Barplot_Liver. Uses the included feature/K aggregate inputs only."""
    PLOT_FEATURES = context.features
    import pandas as pd
    import matplotlib.pyplot as plt
    file_path = PLOT_FEATURES
    df = pd.read_csv(file_path, header=None, names=['Index', 'Marker', 'Value'])
    df = df.head(19)
    df = df.sort_values(by='Value', ascending=True)
    plt.figure(figsize=(9, 7))
    bars = plt.barh(df['Marker'], df['Value'], color='#874AAA')
    plt.xlabel('Importance score', fontsize=19, fontweight='bold')
    plt.ylabel('Feature', fontsize=19, fontweight='bold')
    plt.title('Liver cancer-Feature importance', fontsize=24, fontweight='bold')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=9, fontweight='bold')
    plt.grid(False)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.tick_params(axis='both', width=3, length=8)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontsize(15)
        tick_label.set_fontweight('bold')
    plt.tight_layout()
    plt.savefig('Top_Feature_Barplot_Liver.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main(Path(__file__), globals())
