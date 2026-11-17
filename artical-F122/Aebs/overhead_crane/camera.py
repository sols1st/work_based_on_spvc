"""Orthographic inspection camera, frozen DINO tokens and two-landmark readout."""
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from ..dino_latent.backbone import FrozenDinoV2

HALF_VIEW = 1.8
OFFSET = np.array([-.26, .24])
SWAY_SCALE = .5


def render(z, light, size=176):
    """z contains cart x/y and normalized relative payload x/y."""
    p = np.asarray(z[:2])
    load = p + SWAY_SCALE * np.asarray(z[2:])
    gain = (.85, 1., 1.10)[int(light)]

    def color(rgb):
        return tuple(min(255, round(v * gain)) for v in rgb)

    def pixel(xy):
        x, y = xy
        return ((.5 + x/(2*HALF_VIEW))*size, (.5 - y/(2*HALF_VIEW))*size)

    image = Image.new('RGB', (size, size), color((99, 101, 89)))
    pen = ImageDraw.Draw(image)
    for tick in np.arange(-1.5,1.6,.5):
        for ends in [((tick,-1.8),(tick,1.8)),((-1.8,tick),(1.8,tick))]:
            pen.line([pixel(point) for point in ends], fill=color((128,126,106)))
    def box(center, radius, rgb, fill=True):
        corners=[pixel(center+[-radius,radius]),pixel(center+[radius,-radius])]
        pen.rectangle(corners,fill=color(rgb) if fill else None,outline=color(rgb),width=2)
    box(np.zeros(2),.15,(235,225,155),False)
    pen.line([pixel((-1.8,p[1])), pixel((1.8,p[1]))],fill=color((61,75,76)),width=4)
    pen.line([pixel(p+OFFSET),pixel(p)],fill=color((180,196,196)),width=3)
    box(p,.085,(83,114,119),False)
    pen.line([pixel(p),pixel(load)],fill=color((40,42,44)),width=2)
    box(load,.105,(224,121,47))
    box(load,.055,(157,78,41),False)
    box(p+OFFSET,.080,(55,164,219))
    return np.asarray(image)


class Camera:
    def __init__(self, repo, weights, device):
        self.model = FrozenDinoV2(repo, weights, device)
        self.device = device

    @torch.inference_mode()
    def __call__(self, images):
        rgb = torch.from_numpy(np.stack(images)).to(self.device).permute(0,3,1,2).float()/255
        return self.model.backbone.forward_features(self.model.preprocess(rgb))['x_norm_patchtokens']


class Observation(nn.Module):
    def __init__(self):
        super().__init__()
        self.readout = nn.Sequential(nn.Linear(384,64),nn.GELU(),nn.Linear(64,6))
        self.mix = nn.Linear(4,4)
        self.skew = nn.Parameter(torch.zeros(4,4))
        with torch.no_grad():
            self.mix.weight.copy_(torch.eye(4))
            self.mix.bias.zero_()
        axis = (torch.arange(16)+.5)/16*2-1
        y,x = torch.meshgrid(-axis,axis,indexing='ij')
        self.register_buffer('grid',torch.stack([x,y],dim=-1).reshape(256,2)*HALF_VIEW)
        self.register_buffer('offset',torch.tensor(OFFSET,dtype=torch.float32))

    def matrix(self):
        # Cayley parameterization keeps the learned state proxy invertible.
        skew = self.skew-self.skew.T
        eye = torch.eye(4,device=skew.device)
        return torch.linalg.solve(eye-skew,eye+skew)

    def forward(self, tokens):
        h = self.readout(tokens.float())
        weights = h[:,:,:2].transpose(1,2).softmax(-1)
        offsets = h[:,:,2:].reshape(len(h),256,2,2).transpose(1,2).tanh()*(HALF_VIEW/8)
        points = (weights[:,:,:,None]*(self.grid+offsets)).sum(2)
        p = points[:,0]-self.offset
        z = torch.cat([p,(points[:,1]-p)/SWAY_SCALE],dim=1)
        return self.mix(z),h[:,:,:2].transpose(1,2)

    def decode(self, y):
        return torch.linalg.solve(self.matrix(),y.T).T
