"""Independent pose-image splits. Offline test is development, not final CP."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from Aebs.dino_latent.backbone import FrozenDinoV2
from . import core
from .renderer import RobotRenderer
from .perception import PoseHead


@torch.no_grad()
def predict(model,features):
    device=next(model.parameters()).device
    return torch.cat([model(x.to(device)).cpu() for x in features.split(64)])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--dino-repo',type=Path,required=True)
    p.add_argument('--dino-weights',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--seed',type=int,default=71)
    p.add_argument('--reuse-features',type=Path)
    p.add_argument('--resume',type=Path)
    p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--learning-rate',type=float,default=.001)
    args=p.parse_args(argv)
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True);torch.set_num_threads(args.threads);torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed)
    if args.reuse_features:
        cached=torch.load(args.reuse_features,map_location='cpu',weights_only=False)
        features,states,split=cached['features'],cached['states'],cached['split']
        states,split=np.asarray(states),np.asarray(split)
    else:
        counts=[2400,400,400]
        states=rng.uniform([.25,.25,-1.3],[5.75,3.75,1.3],(sum(counts),3))
        split=np.repeat(np.arange(3),counts)
        renderer=RobotRenderer();backbone=FrozenDinoV2(args.dino_repo,args.dino_weights,args.device)
        features=[];start=time.monotonic()
        for i in range(0,len(states),16):
            images=np.array([renderer.render(x) for x in states[i:i+16]])
            if i==0:
                canvas=Image.new('RGB',(224*4,224*2))
                for j,img in enumerate(images[:8]):canvas.paste(Image.fromarray(img),((j%4)*224,(j//4)*224))
                canvas.save(args.output_dir/'examples.png')
            with torch.no_grad():
                x=torch.tensor(images,device=args.device).permute(0,3,1,2).float()/255
                feat=backbone.backbone.forward_features(backbone.preprocess(x))['x_norm_patchtokens']
            features.append(feat.half().cpu())
            if i%320==0:print(json.dumps(dict(stage='DINO',completed=i,total=len(states),seconds=time.monotonic()-start)),flush=True)
        features=torch.cat(features)
        torch.save(dict(features=features,states=torch.from_numpy(states),split=torch.from_numpy(split)),args.output_dir/'features.pt')
    labels=torch.tensor(core.normalize(states),dtype=torch.float32)
    train=torch.tensor(np.where(split==0)[0]);val=torch.tensor(np.where(split==1)[0]);test=torch.tensor(np.where(split==2)[0])
    model=PoseHead()
    with torch.no_grad():
        f=features[train].float();model.mean.copy_(f.mean((0,1)));model.std.copy_(f.std((0,1)).clamp_min(.1));del f
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=False)
        if saved['format']!='mobile_robot_dino_pose_v1':raise ValueError('invalid resume observer')
        model.load_state_dict(saved['state_dict'])
    model.to(args.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    best=copy.deepcopy(model.state_dict());best_value=float((predict(model,features[val])-labels[val]).square().mean());history=[]
    for epoch in range(args.epochs):
        order=train[torch.randperm(len(train))]
        for ids in order.split(args.batch_size):
            pred=model(features[ids].to(args.device));error=pred-labels[ids].to(args.device)
            loss=error.square().mean()+.05*error.abs().mean()
            optimizer.zero_grad();loss.backward();optimizer.step()
        if (epoch+1)%10==0:
            with torch.no_grad():
                error=predict(model,features[val])-labels[val]
                score=float(error.square().mean())
            if score<best_value:best_value=score;best=copy.deepcopy(model.state_dict())
            entry=dict(epoch=epoch+1,val_mse=score,val_mae=error.abs().mean(0).tolist(),
                       val_joint_q95=float(torch.quantile(error.abs().amax(1),.95)))
            history.append(entry);print(json.dumps(entry),flush=True)
            torch.save(dict(format='mobile_robot_dino_pose_v1',state_dict=best,
                scales=core.SCALE.tolist(),center=core.CENTER.tolist(),seed=args.seed),args.output_dir/'best_observer.pt')
            (args.output_dir/'history.json').write_text(json.dumps(history,indent=2))
    model.load_state_dict(best)
    saved=dict(format='mobile_robot_dino_pose_v1',state_dict=model.state_dict(),
               scales=core.SCALE.tolist(),center=core.CENTER.tolist(),seed=args.seed)
    torch.save(saved,args.output_dir/'observer.pt')
    metrics={}
    for name,ids in [('train',train),('validation',val),('development_test',test)]:
        pred=predict(model,features[ids]);err=pred-labels[ids]
        metrics[name]=dict(n=len(ids),normalized_mae=err.abs().mean(0).tolist(),
                          joint_q95=float(torch.quantile(err.abs().amax(1),.95)),
                          worst=float(err.abs().max()))
        np.savez_compressed(args.output_dir/f'{name}.npz',states=states[ids],prediction=pred.numpy(),errors=err.numpy())
    (args.output_dir/'metrics.json').write_text(json.dumps(metrics,indent=2))
    (args.output_dir/'history.json').write_text(json.dumps(history,indent=2))
    (args.output_dir/'config.json').write_text(json.dumps(dict(task=core.config(),seed=args.seed,
        train=2400,validation=400,development_test=400,epochs=args.epochs,
        source_features=str(args.reuse_features),resume=str(args.resume),device=args.device,
        threads=args.threads,batch_size=args.batch_size,learning_rate=args.learning_rate,
        final_trajectory_calibration=False),indent=2))
    print(json.dumps(metrics),flush=True)


if __name__=='__main__':main()
