本文件夹目前包含以下脚本：
F-score.py：基于 F-score 的特征排序。
FscoretoCSV.py：按照 F-score 排序结果重排原始 CSV。
ReliefF.py：基于 ReliefF 的特征重要性排序。
ReliefFtoCSV.py：按照 ReliefF 排序结果重排原始 CSV。
ANOVA.py：基于方差分析（ANOVA）的特征排序。
ANOVAtoSVM.py：将 ANOVA 排序结果转换为指定格式。
Binomial.py：基于二项检验的二分类特征排序。
Binomial6.py：基于两两类别比较 + Fisher 方法合并 p 值的多分类特征排序。
BinomialtoCSV.py：按照 Binomial 排序结果重排原始 CSV。
SVMtoCSV.py：将 SVM 格式文件转换为 CSV 格式。
这些脚本的用途和输入输出方式可从代码中的 Usage 说明直接查看。

大多数脚本默认输入为 CSV 文件，格式如下：

```csv
class,feature1,feature2,feature3,...
0,0.12,0.35,0.67,...
1,0.22,0.41,0.58,...
