"""Original manuscript Figure S6 plotting functions for pancancer.

Use --list to inspect input requirements and --only to replay selected groups.
Patient-dependent functions read frozen authorized predictions, never fit models.
Public simplified redraws remain available through code/plot_public.py.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'feature_screen'))
from original_plots import ReplayContext, main

def plot_multiclass_roc(context: ReplayContext):
    """Draw six-class ROC curves for context.method using its validation-selected model."""
    predictor = context.predictor
    test_data = context.test_data
    model_dir = str(context.output_dir)
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    label_mapping = {0: 'Control', 1: 'breast cancer' if context.method == 'ReliefF' else 'Breast cancer', 2: 'CRC cancer', 3: 'Gastric cancer', 4: 'Liver cancer', 5: 'Lung cancer'}
    y_true = test_data[predictor.label]
    best_model = predictor.model_best
    print(f'使用的最佳模型是: {best_model}')
    y_proba = predictor.predict_proba(test_data, model=best_model)
    classes = predictor.class_labels
    y_true_bin = label_binarize(y_true, classes=classes)
    roc_data = []
    for (i, class_label) in enumerate(classes):
        sample_weight = y_true_bin[:, i] * 5 + (1 - y_true_bin[:, i])
        (fpr, tpr, _) = roc_curve(y_true_bin[:, i], y_proba.iloc[:, i])
        auc_val = auc(fpr, tpr)
        roc_data.append((label_mapping[class_label], fpr, tpr, auc_val))
    roc_data.sort(key=lambda x: x[3], reverse=True)
    plt.figure(figsize=(10, 8))
    for (label, fpr, tpr, auc_val) in roc_data:
        plt.plot(fpr, tpr, label=f'{label} (AUC = {auc_val:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5000)')
    plt.title(f'Multi-class-{context.method} (by Best Model: {best_model})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{model_dir}', dpi=300)
    plt.show()

def plot_class_indicator_accuracy(context: ReplayContext, dataset='test'):
    """Draw test/train cross-indicator accuracy; this is not a six-class confusion matrix."""
    if dataset not in ('test', 'train'):
        raise ValueError('dataset must be test or train')
    dataset_label = 'Test' if dataset == 'test' else 'Train'
    evaluation_data = context.test_data if dataset == 'test' else context.train_data
    predictor = context.predictor
    data = evaluation_data
    model_dir = str(context.output_dir)
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import label_binarize
    import numpy as np
    import os
    full_label_mapping = {0: 'Control', 1: 'Breast cancer', 2: 'CRC cancer', 3: 'Gastric cancer', 4: 'Liver cancer', 5: 'Lung cancer'}
    labels = [1, 2, 3, 4, 5]
    class_names = [full_label_mapping[i] for i in labels]
    y_true = data[predictor.label]
    y_pred = predictor.predict(data)
    class_labels = predictor.class_labels
    mask = y_true.isin(labels)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    y_true_bin = label_binarize(y_true, classes=class_labels)
    y_pred_bin = label_binarize(y_pred, classes=class_labels)
    acc_matrix = np.zeros((len(labels), len(labels)))
    for (i, true_class) in enumerate(labels):
        for (j, pred_class) in enumerate(labels):
            y_true_binary = (y_true == true_class).astype(int)
            y_pred_binary = (y_pred == pred_class).astype(int)
            acc = accuracy_score(y_true_binary, y_pred_binary)
            acc_matrix[i, j] = acc
    annot = np.array([[f'{acc_matrix[i, j]:.4f}' for j in range(len(labels))] for i in range(len(labels))])
    (fig, ax) = plt.subplots(figsize=(8, 7))
    sns.heatmap(acc_matrix, annot=annot, fmt='', cmap='OrRd', xticklabels=[s.replace(' ', '\n') for s in class_names], yticklabels=[s.replace(' ', '\n') for s in class_names], cbar_kws={'label': 'ACC'}, linewidths=0.0, ax=ax)
    ax.set_xlabel('Prediction', fontsize=15, fontweight='bold', labelpad=10)
    ax.set_ylabel('Reference', fontsize=15, fontweight='bold', labelpad=10)
    ax.set_title(dataset_label, fontsize=19, fontweight='bold', pad=10)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, fontweight='bold', rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, fontweight='bold', rotation=0)
    ax.tick_params(axis='both', length=0)
    for text in ax.texts:
        text.set_fontsize(12)
        text.set_fontweight('bold')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10, width=1.5)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight('bold')
    cbar.ax.yaxis.label.set_fontsize(12)
    cbar.ax.yaxis.label.set_fontweight('bold')
    cbar.ax.yaxis.label.set_rotation(90)
    for (_, spine) in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    plt.tight_layout()
    os.makedirs(model_dir, exist_ok=True)
    plt.savefig(os.path.join(model_dir, f'per_class_accuracy_matrix_{dataset_label}.png'), dpi=300)
    plt.show()

def plot_cancer_confusion_matrix(context: ReplayContext):
    """Draw confusion_matrix_cancer_only_final. Requires authorized patient inputs."""
    predictor = context.predictor
    test_data = context.test_data
    model_dir = str(context.output_dir)
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, accuracy_score
    import numpy as np
    import os
    full_label_mapping = {0: 'Control', 1: 'Breast cancer', 2: 'CRC cancer', 3: 'Gastric cancer', 4: 'Liver cancer', 5: 'Lung cancer'}
    labels = [1, 2, 3, 4, 5]
    class_names = [full_label_mapping[i] for i in labels]
    y_true = test_data[predictor.label]
    best_model = predictor.model_best
    print(f'使用的最佳模型是: {best_model}')
    y_pred = predictor.predict(test_data, model=best_model)
    mask = y_true.isin(labels)
    y_true_cancer = y_true[mask]
    y_pred_cancer = y_pred[mask]
    cm_norm = confusion_matrix(y_true, y_pred, labels=list(range(6)), normalize='true')[1:, 1:]
    acc = accuracy_score(y_true_cancer, y_pred_cancer)
    annot = np.array([[f'{cm_norm[i, j]:.2f}' for j in range(cm_norm.shape[1])] for i in range(cm_norm.shape[0])])
    (fig, ax) = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_norm, annot=annot, fmt='', cmap='OrRd', xticklabels=[s.replace(' ', '\n') for s in class_names], yticklabels=[s.replace(' ', '\n') for s in class_names], cbar_kws={'label': 'Freq'}, linewidths=0.0, ax=ax)
    ax.set_xlabel('Prediction', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reference', fontsize=12, fontweight='bold')
    ax.set_title(f'Test cancer-type accuracy = {acc:.2f}', fontsize=14, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10, fontweight='bold', rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10, fontweight='bold', rotation=0)
    for text in ax.texts:
        text.set_fontsize(12)
        text.set_fontweight('bold')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10, width=1.5)
    cbar.ax.yaxis.label.set_fontsize(12)
    cbar.ax.yaxis.label.set_fontweight('bold')
    cbar.ax.yaxis.label.set_rotation(90)
    for (_, spine) in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    plt.tight_layout()
    os.makedirs(model_dir, exist_ok=True)
    plt.savefig(os.path.join(model_dir, 'confusion_matrix_cancer_only_final.png'), dpi=300)
    plt.show()

def plot_feature_family_composition(context: ReplayContext):
    """Draw pancancer-ReliefF-Proportion of Biomaker. Uses the included feature/K aggregate inputs only."""
    PLOT_FEATURES = context.features
    import pandas as pd
    import matplotlib.pyplot as plt
    file_path = PLOT_FEATURES
    df = pd.read_csv(file_path, header=None, names=['Index', 'Marker', 'Value'])
    df = df.head(20)

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
    colors = ['#fc8d5c', '#fcae62', '#fddaa6', '#feffc0', '#fa9a99']
    (fig, ax) = plt.subplots(figsize=(10, 8))
    (wedges, texts, autotexts) = ax.pie(type_counts, labels=None, autopct='%1.1f%%', startangle=90, colors=colors, pctdistance=0.75, textprops={'fontsize': 18, 'weight': 'bold'})
    plt.legend(type_counts.index, bbox_to_anchor=(0.93, 0.5), loc='center left', frameon=False, borderaxespad=0.3, prop={'family': 'sans-serif', 'size': 17, 'weight': 'bold'})
    plt.title('Pancaner-Proportion of biomaker (top 20)', fontsize=24, fontweight='bold', y=0.94)
    plt.tight_layout()
    plt.savefig('pancancer-ReliefF-Proportion of Biomaker.png', dpi=300)
    plt.show()

def plot_method_feature_count_curve(context: ReplayContext):
    """Draw the original Fscore/ANOVA/Binomial K curve with the method-specific label."""
    method_label = {'Fscore': 'F-score', 'ANOVA': 'ANOVA', 'Binomial': 'Binomial'}[context.method]
    PLOT_CURVE = context.curve
    import matplotlib.pyplot as plt
    import pandas as pd
    auc_df = pd.read_csv(PLOT_CURVE)
    plt.figure(figsize=(10, 6))
    plt.plot(auc_df['Num_Features'], auc_df['Train_AUC'], label='Train AUC', marker='o')
    plt.plot(auc_df['Num_Features'], auc_df['Validation_AUC'], label='Validation AUC', marker='s')
    plt.xlabel('Number of Features')
    plt.ylabel('AUC Score')
    plt.title(f'pan cancer 6 – {method_label}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'AUC_vs_FeatureCount_{context.method}6.png', dpi=300)
    plt.show()

def plot_relieff_feature_count_curve(context: ReplayContext):
    """Draw AUC_vs_FeatureCount_ReliefF_pan6. Uses the included feature/K aggregate inputs only."""
    PLOT_CURVE = context.curve
    import matplotlib.pyplot as plt
    import pandas as pd
    auc_df = pd.read_csv(PLOT_CURVE)
    plt.figure(figsize=(9, 7))
    plt.plot(auc_df['Num_Features'], auc_df['Train_AUC'], label='Train AUC', marker='o', linewidth=2, markersize=7, color='#ad0000')
    plt.plot(auc_df['Num_Features'], auc_df['Validation_AUC'], label='Validation AUC', marker='s', linewidth=2, markersize=7, color='#fa9a99')
    plt.xlabel('Number of Features', fontsize=19, fontweight='bold')
    plt.ylabel('AUC Score', fontsize=19, fontweight='bold')
    plt.title('Pan cancer-Feature selection', fontsize=24, fontweight='bold')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, prop={'family': 'sans-serif', 'size': 15, 'weight': 'bold'})
    plt.grid(True)
    plt.axvline(x=20, color='gray', linestyle='--', linewidth=2)
    xticks = sorted(set(plt.xticks()[0].tolist() + [20]))
    plt.xticks(xticks)
    ax = plt.gca()
    for label in ax.get_xticklabels():
        if label.get_text() == '20':
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
    plt.savefig('AUC_vs_FeatureCount_ReliefF_pan6.png', dpi=300)
    plt.show()

def plot_relieff_validation_feature_count(context: ReplayContext):
    """Draw FeatureCount-pink. Uses the included feature/K aggregate inputs only."""
    PLOT_CURVE = context.curve
    import matplotlib.pyplot as plt
    import pandas as pd
    auc_df = pd.read_csv(PLOT_CURVE)
    bg_color = '#fcedeb'
    plt.rcParams['figure.facecolor'] = bg_color
    plt.rcParams['axes.facecolor'] = bg_color
    plt.rcParams['savefig.facecolor'] = bg_color
    plt.rcParams['savefig.transparent'] = False
    plt.figure(figsize=(9, 7))
    plt.plot(auc_df['Num_Features'], auc_df['Train_AUC'], label='Train AUC', marker='o', linewidth=2, markersize=7, color='#82a5d6')
    plt.plot(auc_df['Num_Features'], auc_df['Validation_AUC'], label='Validation AUC', marker='s', linewidth=2, markersize=7, color='#bf4f6f')
    plt.xlabel('Number of Features', fontsize=19, fontweight='bold')
    plt.ylabel('AUC Score', fontsize=19, fontweight='bold')
    plt.title('Feature selection', fontsize=24, fontweight='bold')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, prop={'family': 'sans-serif', 'size': 15, 'weight': 'bold'})
    plt.grid(False)
    plt.axvline(x=20, color='gray', linestyle='--', linewidth=2)
    xticks = sorted(set(plt.xticks()[0].tolist() + [20]))
    plt.xticks(xticks)
    ax = plt.gca()
    for label in ax.get_xticklabels():
        if label.get_text() == '20':
            label.set_color('red')
    ax.set_xlim(left=0, right=max(auc_df['Num_Features']) + 5)
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.tick_params(axis='both', width=3, length=8)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontsize(15)
        tick_label.set_fontweight('bold')
    plt.tight_layout()
    plt.savefig('FeatureCount-pink.png', dpi=300, facecolor=bg_color)
    plt.show()


if __name__ == '__main__':
    main(Path(__file__), globals())
