"""Original manuscript Figure S3 plotting functions for CRC_cancer.

Use --list to inspect input requirements and --only to replay selected groups.
Patient-dependent functions read frozen authorized predictions, never fit models.
Public simplified redraws remain available through code/plot_public.py.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'feature_screen'))
from original_plots import ReplayContext, main

def plot_feature_family_counts(context: ReplayContext):
    """Draw CRC-ReliefF-Barplot-BiomarkerTypes_NoGrid. Uses the included feature/K aggregate inputs only."""
    PLOT_FEATURES = context.features
    import pandas as pd
    import matplotlib.pyplot as plt
    file_path = PLOT_FEATURES
    df = pd.read_csv(file_path, header=None, names=['Index', 'Marker', 'Value'])
    df = df.head(121)

    def classify_marker(marker):
        if 'GWMD' in marker:
            return 'GWMD'
        elif 'TMD' in marker:
            return 'TMD'
        elif 'GWFP' in marker:
            return 'GWFP'
        elif 'flen' in marker:
            return 'flen'
        elif 'MOTIF' in marker:
            return 'MOTIF'
        elif 'CNA' in marker:
            return 'CNA'
        else:
            return 'Other'
    df['Type'] = df['Marker'].apply(classify_marker)
    type_counts = df['Type'].value_counts().sort_values(ascending=True)
    custom_colors = {'TMD': '#f1a562', 'GWMD': '#88cdea', 'GWFP': '#7570b3', 'flen': '#f79395', 'MOTIF': '#468c74', 'CNA': '#cd4f39', 'Other': '#a6761d'}
    (fig, ax) = plt.subplots(figsize=(9, 6.5))
    bars = ax.barh(type_counts.index, type_counts.values, color=[custom_colors.get(t, 'black') for t in type_counts.index])
    for bar in bars:
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f'{int(bar.get_width())}', va='center', fontsize=15, fontweight='bold')
    ax.set_xlabel('Feature Count', fontsize=19, fontweight='bold')
    ax.set_title('CRC-Biomarker type distribution (Top 121)', fontsize=24, fontweight='bold')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.yticks(fontweight='bold')
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.tick_params(axis='both', width=3, length=8)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontsize(15)
        tick_label.set_fontweight('bold')
    plt.tight_layout()
    plt.savefig('CRC-ReliefF-Barplot-BiomarkerTypes_NoGrid.png', dpi=300)
    plt.show()

def plot_feature_count_curve(context: ReplayContext):
    """Draw AUC_vs_FeatureCount_ReliefF. Uses the included feature/K aggregate inputs only."""
    PLOT_CURVE = context.curve
    import matplotlib.pyplot as plt
    import pandas as pd
    auc_df = pd.read_csv(PLOT_CURVE)
    plt.figure(figsize=(9, 7))
    plt.plot(auc_df['Num_Features'], auc_df['Train_AUC'], label='Train AUC', marker='o', linewidth=2, markersize=7, color='#9370db')
    plt.plot(auc_df['Num_Features'], auc_df['Validation_AUC'], label='Validation AUC', marker='s', linewidth=2, markersize=7, color='#f1787a')
    plt.xlabel('Number of Features', fontsize=19, fontweight='bold')
    plt.ylabel('AUC Score', fontsize=19, fontweight='bold')
    plt.title('Colorectal cancer-Feature selection', fontsize=24, fontweight='bold')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, prop={'family': 'sans-serif', 'size': 15, 'weight': 'bold'})
    plt.grid(True)
    plt.axvline(x=121, color='gray', linestyle='--', linewidth=2)
    xticks = sorted(set(plt.xticks()[0].tolist() + [121]))
    plt.xticks(xticks)
    ax = plt.gca()
    for label in ax.get_xticklabels():
        text = label.get_text()
        if text == '121':
            label.set_x(label.get_position()[0])
            label.set_horizontalalignment('right')
            label.set_color('red')
            label.set_bbox({'facecolor': 'white', 'alpha': 0, 'edgecolor': 'none'})
        elif text == '150':
            label.set_x(label.get_position()[0])
            label.set_horizontalalignment('left')
            label.set_bbox({'facecolor': 'white', 'alpha': 0, 'edgecolor': 'none'})
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlim(left=0, right=max(auc_df['Num_Features']) + 5)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.tick_params(axis='both', width=3, length=8)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontsize(15)
        tick_label.set_fontweight('bold')
    plt.tight_layout()
    plt.savefig('AUC_vs_FeatureCount_ReliefF.png', dpi=300)
    plt.show()

def plot_classifier_score_strips(context: ReplayContext):
    """Draw Figure S3C patient score strips with the original jitter and layout; authorized predictions and splits required."""
    data = context.data
    model_dir = str(context.output_dir)
    output_dir = 'ScoreStripPlots'
    TabularPredictor = context.predictor_api
    train_test_split = context.locked_split
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import os
    from matplotlib.patches import Rectangle
    predictor = TabularPredictor.load(model_dir)
    raw_models = ['XGBoost', 'RandomForest', 'LightGBM', 'NeuralNetFastAI', 'WeightedEnsemble_L2']
    model_name_map = {m: m.replace('', '') for m in raw_models}
    (train_data, test_data) = train_test_split(data, test_size=0.2, stratify=data['class'], random_state=42)
    sets = {'Train': train_data, 'Test': test_data}
    output_path = os.path.join(model_dir, output_dir)
    os.makedirs(output_path, exist_ok=True)
    all_plot_data = []
    for model_name in raw_models:
        for (set_name, df) in sets.items():
            try:
                y_true = df['class']
                y_proba = predictor.predict_proba(df, model=model_name)
                y_score = y_proba.iloc[:, 1] if isinstance(y_proba, pd.DataFrame) else y_proba
                all_plot_data.extend([{'Model': model_name_map[model_name], 'Score': score, 'TrueLabel': true_label, 'Set': set_name} for (score, true_label) in zip(y_score, y_true)])
            except Exception as e:
                print(f'Model {model_name} failed on {set_name}: {e}')
    plot_df = pd.DataFrame(all_plot_data)
    (fig, axes) = plt.subplots(1, 2, figsize=(18, 6.5), sharey=False)
    for (ax, set_name) in zip(axes, ['Train', 'Test']):
        subset = plot_df[plot_df['Set'] == set_name]
        sns.stripplot(data=subset, x='Score', y='Model', orient='h', hue='TrueLabel', palette={0: 'deepskyblue', 1: 'crimson'}, dodge=True, jitter=0.15, size=6, ax=ax)
        ax.axvline(0.5, color='black', linestyle='--', linewidth=2)
        ax.set_xlabel('Classifier Score', fontsize=19)
        ax.tick_params(axis='y', labelsize=10, labelrotation=0)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        if set_name == 'Train':
            ax.set_ylabel('')
        else:
            ax.set_ylabel('')
        ax.set_xlim(-0.05, 1.03)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(axis='y', rotation=0)
        ax.get_legend().remove()
        for spine in ax.spines.values():
            spine.set_linewidth(3)
        ax.tick_params(axis='both', width=3, length=8)
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontsize(17)
            tick_label.set_fontweight('bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'ClassifierScore_ByModel.png'), dpi=300)
    plt.show()


if __name__ == '__main__':
    main(Path(__file__), globals())
