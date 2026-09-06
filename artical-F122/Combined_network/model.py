import torch
import torch.nn as nn
from torch.distributions import Normal

#####################################
# X-Plane11
# ################################### 
class ControllBN(nn.Module):
    def __init__(self, layer_sizes=[128,32,1]):
        super().__init__()
        layers = []
        for i in range(1, len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            layers.append(nn.BatchNorm1d(layer_sizes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)


class End2EndNet(nn.Module):
    def __init__(self, gen_net, controller_layer_sizes=[128,32,1]):
        super().__init__()
        self.gen_net = gen_net
        for param in self.gen_net.parameters():
            param.requires_grad = False
        
        self.controller_net = ControllBN(controller_layer_sizes)
        
    def forward(self, z, ny):
        with torch.no_grad():
            gen_out = self.gen_net(z, ny)
        gen_out = gen_out.view(gen_out.size(0), -1)
        phi = self.controller_net(gen_out)
        return phi
    

#######################################
# Aebs
# #####################################
class SubNet(nn.Module):
    def __init__(self, layer_sizes=[1024,256,64,1]):
        super().__init__()
        layers = []
        for i in range(1, len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            layers.append(nn.LayerNorm(layer_sizes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        out = self.network(x)
        return out

class CombinedPolicyNetwork(nn.Module):
    def __init__(self, mlp_extractor, action_net):
        super(CombinedPolicyNetwork, self).__init__()
        self.mlp_extractor = mlp_extractor
        self.action_net = action_net

    def forward(self, obs):
        latent_pi = self.mlp_extractor(obs)
        action_params = self.action_net(latent_pi)
        
        return action_params
    
class AebsEnd2EndNet(nn.Module):
    def __init__(self, gen_net, state_layer_sizes, mlp_extractor, action_net):
        super().__init__()
        self.gen_net = gen_net
        for p in self.gen_net.parameters():
            p.requires_grad = False
        self.state_net = SubNet(state_layer_sizes)
        for p in self.state_net.parameters():
            p.requires_grad = False
        self.controller_net = CombinedPolicyNetwork(mlp_extractor, action_net)
    def forward(self, z, s):
        with torch.no_grad():
            d = s[:,0].unsqueeze(1)
            v = s[:,1].unsqueeze(1)
            img = self.gen_net(z, d)
            img_flat = img.view(img.size(0), -1)
            state = self.state_net(img_flat)
        x = torch.cat([state, v], dim=1)
        acc = self.controller_net(x)
        return acc
