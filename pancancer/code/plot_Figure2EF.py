"""Regenerate formal six-class E/F from repository aggregate inputs."""
from pathlib import Path
import sys
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'fig2'))
import fig2
if __name__=='__main__':
    out=R/'pancancer/figures/public_reproduced';out.mkdir(parents=True,exist_ok=True)
    fig2.configure('Arial');data=fig2.read_inputs();fig2.validate_data(data)
    for panel in 'EF':fig2.save_pdf(fig2.make_single(panel,data),out/(fig2.STEMS[panel]+'.pdf'))
    print('Generated formal none six-class Figure2E/F')
