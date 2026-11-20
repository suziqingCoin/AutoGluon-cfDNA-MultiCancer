"""Original manuscript Figure S5 plotting functions for Gastric_cancer.

Use --list to inspect input requirements and --only to replay selected groups.
Patient-dependent functions read frozen authorized predictions, never fit models.
Public simplified redraws remain available through code/plot_public.py.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'feature_screen'))
from original_plots import ReplayContext, main

def plot_top_feature_scores(context: ReplayContext):
    """Draw Top_Feature_Barplot_Gastric. Uses the included feature/K aggregate inputs only."""
    PLOT_FEATURES = context.features
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    df = pd.read_csv(PLOT_FEATURES, header=None, names=['Index', 'Marker', 'Value'])
    df = df.sort_values('Value', ascending=False, kind='stable').reset_index(drop=True)
    assert len(df) == 10 and df['Value'].notna().all()
    (fig, ax) = plt.subplots(figsize=(9, 7))
    positions = np.arange(len(df))
    ax.barh(positions, df['Value'], height=0.8, color='#1f77b4')
    ax.set_yticks(positions, df['Marker'], fontsize=13)
    ax.set_ylim(len(df) - 0.45, -0.55)
    xmax = np.ceil(df['Value'].max() * 1.05 / 0.01) * 0.01
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_xlabel('Importance score', fontsize=19)
    ax.set_title('Gastric cancer-Feature importance', fontsize=24, pad=15)
    ax.tick_params(axis='both', width=3, length=8)
    ax.tick_params(axis='x', labelsize=15)
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.grid(False)
    fig.tight_layout()
    assert len(fig.axes) == 1 and len(ax.patches) == 10
    assert df['Value'].is_monotonic_decreasing
    fig.savefig('Top_Feature_Barplot_Gastric.png', dpi=300)
    plt.show()

def plot_feature_count_improvement(context: ReplayContext):
    """Draw DeltaAUC_and_Trend_Highlight10. Uses the included feature/K aggregate inputs only."""
    PLOT_CURVE = context.curve
    import matplotlib.pyplot as plt
    import pandas as pd
    auc_df = pd.read_csv(PLOT_CURVE)
    auc_df['Delta_Val_AUC'] = auc_df['Validation_AUC'].diff()
    (fig, ax1) = plt.subplots(figsize=(12, 7))
    bars = ax1.bar(auc_df['Num_Features'], auc_df['Delta_Val_AUC'], color='#9370db', edgecolor='purple')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1.8)
    ax1.set_xlabel('Number of Features', fontsize=19, fontweight='bold')
    ax1.set_ylabel('Δ AUC (Validation)', fontsize=19, color='black', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='purple')
    plt.axvline(x=10, color='#d3521f', linestyle='--', linewidth=2)
    xticks = sorted(set(plt.xticks()[0].tolist() + [10]))
    plt.xticks(xticks)
    ax = plt.gca()
    for label in ax.get_xticklabels():
        text = label.get_text()
        if text == '10':
            label.set_color('#d3521f')
            label.set_bbox({'facecolor': 'white', 'alpha': 0.9, 'edgecolor': 'none'})
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.tick_params(axis='both', width=3, length=8)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontsize(15)
        tick_label.set_fontweight('bold')
    ax2 = ax1.twinx()
    ax2.plot(auc_df['Num_Features'], auc_df['Validation_AUC'], color='#1f77b4', marker='o', markersize=8, linewidth=2.5, label='Validation AUC')
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
    plt.title('Gastric cancer-Feature selection', fontsize=24, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('DeltaAUC_and_Trend_Highlight10.png', dpi=300)
    plt.show()

def plot_model_metric_heatmap(context: ReplayContext, dataset='test'):
    """Draw the original test/train metric heatmap from authorized frozen predictions."""
    if dataset not in ('test', 'train'):
        raise ValueError('dataset must be test or train')
    dataset_label = 'Test' if dataset == 'test' else 'Train'
    evaluation_data = context.test_data if dataset == 'test' else context.train_data
    data = context.data
    model_dir = str(context.output_dir)
    TabularPredictor = context.predictor_api
    train_test_split = context.locked_split
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, recall_score, precision_score
    predictor = TabularPredictor.load(model_dir)
    models = ['XGBoost', 'RandomForest', 'LightGBM', 'NeuralNetFastAI', 'WeightedEnsemble_L2']
    model_name_map = {m: m.replace('', '') for m in models}
    (train_data, test_data) = train_test_split(data, test_size=0.2, stratify=data['class'], random_state=42)
    results = []
    for model_name in models:
        try:
            y_true = evaluation_data['class']
            y_proba = predictor.predict_proba(evaluation_data, model=model_name)
            y_score = y_proba.iloc[:, 1] if isinstance(y_proba, pd.DataFrame) else y_proba
            y_pred = predictor.predict(evaluation_data, model=model_name)
            auc = roc_auc_score(y_true, y_score)
            auprc = average_precision_score(y_true, y_score)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            results.append([model_name_map[model_name], auprc, auc, acc, f1, recall, precision])
        except Exception as e:
            print(f'{model_name} failed: {e}')
            results.append([model_name_map[model_name]] + [np.nan] * 6)
    df = pd.DataFrame(results, columns=['Model', 'AUPRC', 'AUROC', 'ACC', 'F1', 'Recall', 'Precision'])
    df.set_index('Model', inplace=True)
    plt.figure(figsize=(12, 8))
    sns.heatmap(df, annot=True, cmap='Blues', fmt='.4f', linewidths=0.5, annot_kws={'size': 14})
    cbar = plt.gca().collections[0].colorbar
    cbar.ax.tick_params(labelsize=14, width=2)
    for tick_label in cbar.ax.get_yticklabels():
        tick_label.set_fontweight('bold')
    plt.title(f'{dataset_label} - Performance Metrics Heatmap', fontsize=24, fontweight='bold')
    plt.ylabel('')
    plt.xticks(fontsize=15, fontweight='bold')
    plt.yticks(fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{model_dir}/model_performance_heatmap_{dataset}.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main(Path(__file__), globals())
