from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from protocol import *
from fold import valid
from autogluon.tabular import TabularPredictor
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
import xgboost,skrebate,psutil,openpyxl
assert sum(COUNTS.values())==8488 and len(METHODS)==4 and len(MODELS)==5
print('PASS: Nested LOOCV imports, 8488 fold counts, 4 methods, 5 models; no training')
