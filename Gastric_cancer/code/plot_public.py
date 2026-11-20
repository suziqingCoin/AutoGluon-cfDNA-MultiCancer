from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'feature_screen'))
from plot_public import main
if __name__=='__main__':main(Path(__file__).resolve().parents[1])
