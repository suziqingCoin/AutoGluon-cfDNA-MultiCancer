"""Original manuscript Figure S1 plotting functions for Lung_cancer.

Use --list to inspect input requirements and --only to replay selected groups.
Patient-dependent functions read frozen authorized predictions, never fit models.
Public simplified redraws remain available through code/plot_public.py.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'feature_screen'))
from original_plots import ReplayContext, main

def plot_feature_count_curve(context: ReplayContext):
    """Draw AUC_vs_FeatureCount_ReliefF. Uses the included feature/K aggregate inputs only."""
    PLOT_CURVE = context.curve
    import matplotlib.pyplot as plt
    import pandas as pd
    auc_df = pd.read_csv(PLOT_CURVE)
    plt.figure(figsize=(9, 7))
    plt.plot(auc_df['Num_Features'], auc_df['Train_AUC'], label='Train AUC', marker='o', linewidth=2, markersize=7)
    plt.plot(auc_df['Num_Features'], auc_df['Validation_AUC'], label='Validation AUC', marker='s', linewidth=2, markersize=7)
    plt.xlabel('Number of Features', fontsize=19, fontweight='bold')
    plt.ylabel('AUC Score', fontsize=19, fontweight='bold')
    plt.title('Lung cancer-Feature selection', fontsize=24, fontweight='bold')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, prop={'family': 'sans-serif', 'size': 15, 'weight': 'bold'})
    plt.grid(True)
    plt.axvline(x=149, color='gray', linestyle='--', linewidth=2)
    xticks = sorted(set(plt.xticks()[0].tolist() + [149]))
    plt.xticks(xticks)
    ax = plt.gca()
    for label in ax.get_xticklabels():
        if label.get_text() == '149':
            label.set_color('red')
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

def plot_feature_family_composition(context: ReplayContext):
    """Draw Lung-ReliefF-Proportion of Biomaker_Optimized. Uses the included feature/K aggregate inputs only."""
    PLOT_FEATURES = context.features
    import pandas as pd
    import matplotlib.pyplot as plt
    file_path = PLOT_FEATURES
    df = pd.read_csv(file_path, header=None, names=['Index', 'Marker', 'Value'])
    df = df.head(149)

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
    type_counts = df['Type'].value_counts()
    colors = plt.cm.Paired(range(len(type_counts)))
    (fig, ax) = plt.subplots(figsize=(10, 8))
    (wedges, texts, autotexts) = ax.pie(type_counts, labels=None, autopct='%1.1f%%', startangle=90, colors=colors, pctdistance=0.75, textprops={'fontsize': 19, 'weight': 'bold'})
    plt.legend(type_counts.index, bbox_to_anchor=(0.93, 0.5), loc='center left', frameon=False, borderaxespad=0.3, prop={'family': 'sans-serif', 'size': 17, 'weight': 'bold'})
    plt.title('Lung caner-Proportion of biomaker (top 149)', fontsize=24, fontweight='bold', y=0.94)
    plt.tight_layout()
    plt.savefig('Lung-ReliefF-Proportion of Biomaker_Optimized.png', dpi=300)
    plt.show()

def plot_test_roc(context: ReplayContext):
    """Draw ReliefF. Requires authorized patient inputs."""
    data = context.data
    model_dir = str(context.output_dir)
    TabularPredictor = context.predictor_api
    train_test_split = context.locked_split
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    ReliefF_predictor = TabularPredictor.load(model_dir)
    models = ['XGBoost', 'RandomForest', 'LightGBM', 'NeuralNetFastAI', 'WeightedEnsemble_L2']
    auc_scores = []
    plt.figure(figsize=(10, 7))
    (train_data, test_data) = train_test_split(data, test_size=0.2, random_state=42, stratify=data['class'])
    y_true = test_data['class']
    plot_data = []
    for model_name in models:
        model = ReliefF_predictor._trainer.load_model(model_name)
        y_prob = model.predict_proba(test_data)
        (fpr, tpr, thresholds) = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        auc_scores.append((model_name, roc_auc))
        clean_name = model_name.replace('', '').replace('_L1', '')
        plot_data.append((roc_auc, fpr, tpr, clean_name))
    plot_data.append((0.5, [0, 1], [0, 1], 'Random'))
    plot_data.sort(reverse=True, key=lambda x: x[0])
    for (roc_auc, fpr, tpr, label) in plot_data:
        if label.lower() == 'random':
            plt.plot(fpr, tpr, lw=4, linestyle='--', color='gray', label=f'{label} (AUC = {roc_auc:.4f})')
        else:
            plt.plot(fpr, tpr, lw=5, label=f'{label} (AUC = {roc_auc:.4f})')
    plt.xlabel('False Positive Rate', fontweight='bold', fontsize=19)
    plt.ylabel('True Positive Rate', fontweight='bold', fontsize=19)
    plt.title('Lung cancer-Test dataset', fontweight='bold', fontsize=24)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.tick_params(axis='both', width=3, length=8)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontsize(15)
        tick_label.set_fontweight('bold')
    plt.legend(loc='lower right', frameon=False, prop={'family': 'sans-serif', 'size': 14, 'weight': 'bold'})
    plt.savefig(f'{model_dir}', dpi=300)
    plt.show()

def plot_test_precision_recall(context: ReplayContext):
    """Draw ReliefF_PR. Requires authorized patient inputs."""
    FIGURES = context.figures
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score
    base = FIGURES.parent / 'results/holdout/none/ReliefF'
    metrics = pd.read_csv(FIGURES.parent / 'results/holdout/all_none_evaluation_metrics.csv')
    curves = []
    reference = None
    for model in ['XGBoost', 'RandomForest', 'LightGBM', 'NeuralNetFastAI', 'WeightedEnsemble_L2']:
        data = pd.read_csv(base / f'evaluation_predictions_{model}.csv').sort_values('SampleID')
        assert data['strategy'].eq('none').all() and data['method'].eq('ReliefF').all()
        ids_labels = list(zip(data['SampleID'], data['y_true']))
        if reference is None:
            reference = ids_labels
        else:
            assert ids_labels == reference
        (precision, recall, thresholds) = precision_recall_curve(data['y_true'], data['p_1'])
        ap = average_precision_score(data['y_true'], data['p_1'])
        expected = metrics.loc[metrics['method'].eq('ReliefF') & metrics['model'].eq(model), 'AUPRC'].iloc[0]
        assert np.isclose(ap, expected, rtol=1e-12, atol=1e-12)
        curves.append((model, precision, recall, ap))
    assert len(reference) == 338
    prevalence = float(data['y_true'].mean())
    (fig, ax) = plt.subplots(figsize=(10, 7))
    colors = {'XGBoost': '#1f77b4', 'LightGBM': '#ff7f0e', 'RandomForest': '#2ca02c', 'WeightedEnsemble_L2': '#d62728', 'NeuralNetFastAI': '#9467bd'}
    for (model, precision, recall, ap) in sorted(curves, key=lambda item: item[3], reverse=True):
        ax.step(recall, precision, where='post', color=colors[model], linewidth=5, label=f'{model} (AUPRC = {ap:.4f})')
    ax.axhline(prevalence, color='gray', linestyle='--', linewidth=4, label=f'Random (AUPRC = {prevalence:.4f})')
    ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))
    ax.set_xlabel('Recall', fontsize=19)
    ax.set_ylabel('Precision', fontsize=19)
    ax.set_title('Lung cancer-Test dataset', fontsize=24)
    ax.grid(False)
    ax.tick_params(axis='both', labelsize=15, width=3, length=8)
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.legend(loc='lower left', frameon=False, prop={'family': 'Arial', 'size': 14, 'weight': 'normal'})
    assert all((line.get_linewidth() == 5 for line in ax.lines[:5]))
    assert ax.lines[-1].get_linewidth() == 4
    assert len(ax.lines) == 6
    fig.savefig('ReliefF_PR.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main(Path(__file__), globals())
