# Formal none holdout protocol

Patient-independent class-stratified 80/20 split, seed42. Each five-cancer task uses all 1550 controls and only the corresponding cancers. Six-class targets: Control=0, Breast=1, CRC=2, Gastric=3, Liver=4, Lung=5. Molecular inputs are the original columns after four metadata columns; preserve row/column order. Inner stratified 80/20 split of the 80% training set: seed42 for binary tasks, seed43 for pan6.

No resampling or class/sample weighting. Four inner-training-only rankings Fscore/ANOVA/ReliefF/Binomial. K search uses the exact source XGBoost settings and per-task caps in none_protocol.py, 50 rounds, early stop10; stop after 10 consecutive train-minus-validation AUC gaps >0.1 or 5 consecutive decreases. First strict validation maximum determines K. AutoGluon1.1.1, five model families per method; no bagging/stacking, dynamic_stacking=False, GPU0, CPU2, original default hyperparameters and fitted parameters retained in results/holdout/none/<method>/*.json.

The 20 internal validation candidates are ordered descending AUC/AP/Recall/healthy specificity rounded12 decimals, then ascending K, fixed method order, model name. Selection is locked before Test access. Test candidate metrics are descriptive and never selection input. Feature identity in the formal holdout is the inner-training ranking prefix, retained for the final full-training refit; it is not reranked on Test or all cohort. Holdout AutoGluon feature generation retains the original behavior (unlabeled validation may participate); strict Nested LOOCV explicitly fits feature generators on inner/outer training separately.

Final refit uses AutoGluon1.1.1 advanced refit_full with transformed full training, empty validation, inherited trained hyperparameters; actual base-model fit rows are monitored. Existing interrupted FULL refits are archived/retried. Do not change this version-dependent API casually. Ensemble meta-weights are validation fitted. Binary argmax tie goes to Control.

Optional --training-cv executes original 3-fold outer CV on the 80% training set, each with its own inner holdout. This is separate from full-cohort Nested LOOCV. The public revision does not rerun any completed training.

Conditional Test CIs: patient-aligned class-stratified bootstrap seed42, 2000 replicates, probabilities ordered by SampleID. Healthy specificity also has Clopper–Pearson exact 95% CI. Same cohort and historically viewed Test; no claim of new external validation.
