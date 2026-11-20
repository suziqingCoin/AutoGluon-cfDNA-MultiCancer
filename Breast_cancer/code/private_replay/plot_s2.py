"""Original manuscript Figure S2 plotting functions for Breast_cancer.

Use --list to inspect input requirements and --only to replay selected groups.
Patient-dependent functions read frozen authorized predictions, never fit models.
Public simplified redraws remain available through code/plot_public.py.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'feature_screen'))
from original_plots import ReplayContext, main

def plot_model_auc_comparison(context: ReplayContext):
    """Draw ANOVA_AUC_dotplot. Requires authorized patient inputs."""
    data = context.data
    model_dir = str(context.output_dir)
    TabularPredictor = context.predictor_api
    train_test_split = context.locked_split
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score
    import pandas as pd
    predictor = TabularPredictor.load(model_dir)
    models = ['XGBoost', 'RandomForest', 'LightGBM', 'NeuralNetFastAI', 'WeightedEnsemble_L2']
    (train_data, test_data) = train_test_split(data, test_size=0.2, random_state=42, stratify=data['class'])
    auc_train = []
    auc_val = []
    for model_name in models:
        model = predictor._trainer.load_model(model_name)
        y_train = train_data['class']
        y_val = test_data['class']
        y_prob_train = model.predict_proba(train_data)
        y_prob_val = model.predict_proba(test_data)
        if isinstance(y_prob_train, pd.DataFrame):
            y_prob_train = y_prob_train.iloc[:, 1]
            y_prob_val = y_prob_val.iloc[:, 1]
        auc_tr = roc_auc_score(y_train, y_prob_train)
        auc_te = roc_auc_score(y_val, y_prob_val)
        auc_train.append(auc_tr)
        auc_val.append(auc_te)
    combined = list(zip(models, auc_train, auc_val))
    combined.sort(key=lambda x: x[2], reverse=False)
    model_labels = [x[0].replace('', '') for x in combined]
    auc_train_sorted = [x[1] for x in combined]
    auc_val_sorted = [x[2] for x in combined]
    y_pos = list(range(len(models)))
    (fig, ax) = plt.subplots(figsize=(5.5, 4))
    ax.scatter(auc_train_sorted, y_pos, color='deepskyblue', label='Train', s=80, zorder=3)
    ax.scatter(auc_val_sorted, y_pos, facecolors='none', edgecolors='brown', label='Test', s=80, linewidths=2, zorder=3)
    for y in y_pos:
        ax.hlines(y, 0.5, 1.05, color='lightgray', linestyle='--', linewidth=0.7, zorder=0)
    ax.set_yticks(y_pos)
    ax.set_xlabel('AUC', fontsize=12, fontweight='bold')
    ax.set_title('Breast cancer-Test', fontsize=16, fontweight='bold')
    ax.set_xlim(0.5, 1.05)
    ax.set_ylim(-0.5, len(models) - 0.85 + 0.7)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(axis='both', width=2, length=6)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontsize(10)
        tick_label.set_fontweight('bold')
    ax.set_yticklabels(model_labels, fontsize=11)
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 1.02), frameon=False, prop={'family': 'sans-serif', 'size': 10, 'weight': 'bold'})
    plt.tight_layout()
    plt.savefig(f'{model_dir}_AUC_dotplot.png', dpi=300)
    plt.show()

def plot_feature_family_counts(context: ReplayContext):
    """Draw Breast-ANOVA-Barplot-BiomarkerTypes_NoGrid. Uses the included feature/K aggregate inputs only."""
    PLOT_FEATURES = context.features
    import pandas as pd
    import matplotlib.pyplot as plt
    file_path = PLOT_FEATURES
    df = pd.read_csv(file_path, header=None, names=['Index', 'Marker', 'Value'])
    df = df.head(151)

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
    custom_colors = {'TMD': '#f1a562', 'GWMD': '#88cdea', 'GWFP': '#7570b3', 'flen': '#e7298a', 'MOTIF': '#468c74', 'CNA': '#cd4f39', 'Other': '#a6761d'}
    (fig, ax) = plt.subplots(figsize=(11, 7))
    bars = ax.barh(type_counts.index, type_counts.values, color=[custom_colors.get(t, 'black') for t in type_counts.index])
    for bar in bars:
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f'{int(bar.get_width())}', va='center', fontsize=15, fontweight='bold')
    ax.set_xlabel('Feature Count', fontsize=19, fontweight='bold')
    ax.set_title('Breast cancer-Biomarker type distribution (Top 151)', fontsize=24, fontweight='bold')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.tick_params(axis='both', width=3, length=8)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontsize(15)
        tick_label.set_fontweight('bold')
    plt.tight_layout()
    plt.savefig('Breast-ANOVA-Barplot-BiomarkerTypes_NoGrid.png', dpi=300)
    plt.show()

def plot_feature_count_improvement(context: ReplayContext):
    """Draw DeltaAUC_and_Trend_Highlight151. Uses the included feature/K aggregate inputs only."""
    PLOT_CURVE = context.curve
    import matplotlib.pyplot as plt
    import pandas as pd
    auc_df = pd.read_csv(PLOT_CURVE)
    auc_df['Delta_Val_AUC'] = auc_df['Validation_AUC'].diff()
    (fig, ax1) = plt.subplots(figsize=(12, 7))
    bars = ax1.bar(auc_df['Num_Features'], auc_df['Delta_Val_AUC'], color='#ed9678', edgecolor='orange')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1.8)
    ax1.set_xlabel('Number of Features', fontsize=19, fontweight='bold')
    ax1.set_ylabel('Δ AUC (Validation)', fontsize=19, color='black', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='orange')
    plt.axvline(x=151, color='#d3521f', linestyle='--', linewidth=2)
    xticks = sorted(set(plt.xticks()[0].tolist() + [151]))
    plt.xticks(xticks)
    ax = plt.gca()
    for label in ax.get_xticklabels():
        text = label.get_text()
        if text == '151':
            label.set_x(label.get_position()[0])
            label.set_horizontalalignment('right')
            label.set_color('#d3521f')
            label.set_bbox({'facecolor': 'white', 'alpha': 0.9, 'edgecolor': 'none'})
        elif text == '120':
            label.set_x(label.get_position()[0])
            label.set_horizontalalignment('left')
            label.set_bbox({'facecolor': 'white', 'alpha': 0.9, 'edgecolor': 'none'})
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.tick_params(axis='both', width=3, length=8)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontsize(15)
        tick_label.set_fontweight('bold')
    ax2 = ax1.twinx()
    ax2.plot(auc_df['Num_Features'], auc_df['Validation_AUC'], color='#1f77b4', marker='o', markersize=6, linewidth=2.5, label='Validation AUC')
    ax2.set_ylabel('Validation AUC', fontsize=19, color='black', fontweight='bold', labelpad=11)
    ax2.tick_params(axis='y', labelcolor='#1f77b4')
    ax2.set_ylim(auc_df['Validation_AUC'].min() - 0.01, auc_df['Validation_AUC'].max() + 0.01)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.tick_params(axis='both', width=3, length=8)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontsize(15)
        tick_label.set_fontweight('bold')
    plt.title('Breast cancer-Feature selection', fontsize=24, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('DeltaAUC_and_Trend_Highlight151.png', dpi=300)
    plt.show()

def plot_model_roc_curves(context: ReplayContext):
    """Draw XGBoost_ROC, RandomForest_ROC, LightGBM_ROC, NeuralNetFastAI_ROC, WeightedEnsemble_L2_ROC. Requires authorized patient inputs."""
    data = context.data
    model_dir = str(context.output_dir)
    output_dir = 'ROC_plots'
    TabularPredictor = context.predictor_api
    train_test_split = context.locked_split
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    import pandas as pd
    import os
    import numpy as np
    predictor = TabularPredictor.load(model_dir)
    models = ['XGBoost', 'RandomForest', 'LightGBM', 'NeuralNetFastAI', 'WeightedEnsemble_L2']
    (train_data, test_data) = train_test_split(data, test_size=0.2, random_state=42, stratify=data['class'])
    output_path = os.path.join(model_dir, output_dir)
    os.makedirs(output_path, exist_ok=True)
    for model_name in models:
        model = predictor._trainer.load_model(model_name)
        (fig, ax) = plt.subplots(figsize=(4, 4))
        for (dataset_name, dataset, color) in zip(['Train', 'Test'], [train_data, test_data], ['blue', 'red']):
            y_true = dataset['class']
            y_proba = model.predict_proba(dataset)
            if isinstance(y_proba, pd.DataFrame):
                y_proba = y_proba.iloc[:, 1]
            (fpr, tpr, _) = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            linestyle = '--' if dataset_name == 'Train' else '-'
            ax.plot(fpr, tpr, label=f'{dataset_name}: {roc_auc:.4f}', color=color, linestyle=linestyle, linewidth=2)
        ticks = np.linspace(0, 1, 5)
        tick_labels = [f'{t:.2f}' for t in ticks]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xticks(np.linspace(0, 1, 9), minor=True)
        ax.set_yticks(np.linspace(0, 1, 9), minor=True)
        ax.tick_params(axis='both', which='major', direction='out', length=6, width=1.2, labelsize=10)
        ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)
        clean_name = model_name.replace('', '')
        ax.set_title(clean_name, fontsize=15, fontweight='bold')
        ax.set_xlabel('FPR', fontsize=12, fontweight='bold')
        ax.set_ylabel('TPR', fontsize=12, fontweight='bold')
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        ax.tick_params(axis='both', width=2, length=6)
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontweight('bold')
        ax.legend(loc='lower right', fontsize=8, frameon=False, prop={'size': 11, 'weight': 'bold'})
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'{model_name}_ROC.png'), dpi=300)
        plt.show()


if __name__ == '__main__':
    main(Path(__file__), globals())
