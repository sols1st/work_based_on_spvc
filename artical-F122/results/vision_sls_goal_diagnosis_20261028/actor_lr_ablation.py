import copy,json
from pathlib import Path
import torch,numpy as np
from stable_baselines3 import PPO
from Combined_network.model import CombinedPolicyNetwork
from Aebs.vision_sls.model import VisionSLSPolicy
from Aebs.vision_sls.run import evaluate
from Aebs.system.env import Aebs
from Aebs.VT.train import VTLearner
from Aebs.VT.verify import VTVerifier
p=Path('results/vision_sls_spvc_20261028_final'); out=Path('results/vision_sls_goal_diagnosis_20261028')
torch.set_num_threads(4); c=torch.load(p/'observation.pt',weights_only=False); x=np.load(p/'observations.npz'); ids=np.load(p/'splits.npz')['train']
res={}
for lr in [.05,.0003,.0001]:
 torch.manual_seed(71); np.random.seed(71)
 ppo=PPO.load(p/'ppo_initial.zip',device='cpu'); policy=VisionSLSPolicy(CombinedPolicyNetwork(ppo.policy.mlp_extractor.policy_net,ppo.policy.action_net),c['distance_scale_m'])
 env=Aebs(.05); policy.set_state_distance_scale(env.std1)
 learner=VTLearner([2,16,8,1],env,2.,4.,.1,1.,.95,p_net=policy,l_model_path=str(p/'sbc.pt'))
 learner.p_optimizer=torch.optim.Adam(policy.controller_net.parameters(),lr=lr)
 verifier=VTVerifier(learner,env,learner.create_bounded_module(learner.l_model),2048,.9,True)
 grid,_,_=verifier.get_unfiltered_grid(env.train_space_split)
 epoch=learner.train_epoch(grid,(env.observation_space.high-env.observation_space.low)/env.train_space_split,lip=.001,batch_size=2048,num_epochs=1,train_fn='p')
 res[str(lr)]=dict(train_loss=epoch,nominal=evaluate(policy,c['distance_scale_m']),train_image=evaluate(policy,c['distance_scale_m'],x['distance_m'][ids],x['observation'][ids]))
 print(json.dumps({str(lr):res[str(lr)]}),flush=True)
(out/'actor_lr_ablation.json').write_text(json.dumps(res,indent=2))
