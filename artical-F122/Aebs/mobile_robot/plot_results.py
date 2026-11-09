"""Export a static, reproducible top-down trajectory figure."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Rectangle
from . import core


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rollout',type=Path,nargs='+',required=True)
    p.add_argument('--labels',nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if len(args.rollout)!=len(args.labels):raise ValueError('one label per rollout')
    fig,axes=plt.subplots(1,len(args.labels),figsize=(6*len(args.labels),4.5),squeeze=False)
    for ax,file,label in zip(axes[0],args.rollout,args.labels):
        data=np.load(file);states=data['states'];bad=core.unsafe(states).any(0)
        ax.add_patch(Circle(core.OBSTACLE,core.EXCLUSION_RADIUS,color='tomato',alpha=.18))
        ax.add_patch(Circle(core.OBSTACLE,core.OBSTACLE_RADIUS,color='slategray'))
        ax.add_patch(Circle(core.GOAL,core.GOAL_RADIUS,color='seagreen',alpha=.3))
        ax.add_patch(Rectangle(core.INITIAL_LOW[:2],*(core.INITIAL_HIGH[:2]-core.INITIAL_LOW[:2]),facecolor='royalblue',alpha=.25))
        ax.add_patch(Rectangle(core.LOW[:2],*(core.HIGH[:2]-core.LOW[:2]),fill=False,linestyle='--',edgecolor='gray'))
        for i in np.linspace(0,states.shape[1]-1,min(30,states.shape[1]),dtype=int):
            ax.plot(states[:,i,0],states[:,i,1],color='crimson' if bad[i] else '#2364AA',alpha=.4,lw=.9)
        ax.set(xlim=(0,6),ylim=(0,4),xlabel='x (m)',ylabel='y (m)',title=label+' | unsafe: '+str(int(bad.sum()))+'/'+str(len(bad)))
        ax.set_aspect('equal');ax.grid(alpha=.15)
    fig.suptitle('3D mobile robot: (x, y, heading), 120 control steps')
    fig.tight_layout();args.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(args.output,dpi=180);plt.close(fig)


if __name__=='__main__':main()
