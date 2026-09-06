import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Combined_network.model import AebsEnd2EndNet
from Aebs.VT.utils import triangular, martingale_loss, MLP
from cGAN.taxi_models_and_data import AebsMLPGenerator
from Combined_network.model import AebsEnd2EndNet
from stable_baselines3 import PPO
from auto_LiRPA import BoundedModule

# -------------------------
# VTLearner
# -------------------------
class VTLearner:
    def __init__(
        self,
        l_model_config,
        env,
        p_lip,
        l_lip,
        eps,
        gamma_decrease,
        reach_prob,
        square_l_output=True,
        l_model_path=None,
        p_model_path=None
    ) -> None:
        self.env = env
        self.eps = float(eps)
        self.gamma_decrease = gamma_decrease
        self.reach_prob = float(reach_prob)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        action_dim = self.env.action_space.shape[0]
        obs_dim = self.env.observation_space.shape[0]

        # l_net
        self.l_model = MLP(l_model_config, activation="tanh", square_output=square_l_output).to(self.device)

        if l_model_path is not None:
            print(f"Loading Lyapunov model from: {l_model_path}")
            l_model_state_dict = torch.load(l_model_path, map_location=self.device)
            self.l_model.load_state_dict(l_model_state_dict)
            print("Lyapunov model loaded successfully!")

        # load End2End
        gen_net = AebsMLPGenerator(4, 1)
        gen_net.load_state_dict(torch.load("./Aebs/cGAN/mlp_supervised_ld4/mlp_supervised.pth"))
        state_layer_sizes = [1024, 256, 64, 1]
        model = PPO.load('./Aebs/controller/best_model/best_model.zip')
        policy = model.policy
        mlp_extractor = policy.mlp_extractor.policy_net
        action_net = policy.action_net
        p_net = AebsEnd2EndNet(gen_net, state_layer_sizes, mlp_extractor, action_net)
        p_net.state_net.load_state_dict(torch.load("./Aebs/controller/state_net_trained.pth"))
        p_net.eval()
        p_net.to(self.device)
        self.net = copy.deepcopy(p_net)
        for param in self.net.parameters():
            param.requires_grad = False
        if p_model_path is not None:
            p_net.load_state_dict(torch.load(p_model_path))
        self.p_net = p_net

        # Optimizers for l_net and controller_net 
        self.l_optimizer = torch.optim.Adam(self.l_model.parameters(), lr=3e-3)
        self.p_optimizer = torch.optim.Adam(self.p_net.controller_net.parameters(), lr=5e-2)

        # Lipschitz target thresholds
        self.l_lip = float(l_lip)
        self.p_lip = float(p_lip)

    def sample_init(self, rng_seed, n):
        num_spaces = len(self.env.init_spaces)
        per_space_n = n // num_spaces
        rng = torch.Generator(device="cpu")
        rng.manual_seed(rng_seed)
        batch = []
        for i in range(num_spaces):
            low = torch.tensor(self.env.init_spaces[i].low, dtype=torch.float32)
            high = torch.tensor(self.env.init_spaces[i].high, dtype=torch.float32)
            shape = (per_space_n, self.env.observation_space.shape[0])
            x = (high - low) * torch.rand(shape, generator=rng) + low
            batch.append(x)
        return torch.cat(batch, dim=0).to(self.device)

    def sample_unsafe(self, rng_seed, n):
        num_spaces = len(self.env.unsafe_spaces)
        per_space_n = n // num_spaces
        rng = torch.Generator(device="cpu")
        rng.manual_seed(rng_seed)
        batch = []
        for i in range(num_spaces):
            low = torch.tensor(self.env.unsafe_spaces[i].low, dtype=torch.float32)
            high = torch.tensor(self.env.unsafe_spaces[i].high, dtype=torch.float32)
            shape = (per_space_n, self.env.observation_space.shape[0])
            x = (high - low) * torch.rand(shape, generator=rng) + low
            batch.append(x)
        return torch.cat(batch, dim=0).to(self.device)

    def train_step_l(self, z, y, lip_coeff, current_delta, clip_grad=5.0):
        device = self.device
        B = y.shape[0]

        # Regional sampling
        init_samples = self.sample_init(13, 256)
        unsafe_samples = self.sample_unsafe(17, 256)

        # Disturb the initial input state
        rng = torch.Generator(device=self.device)
        rng.manual_seed(19)
        s_random = torch.rand(y.shape, generator=rng, device=device) - 0.5
        current_delta = torch.tensor(current_delta, dtype=y.dtype, device=y.device).unsqueeze(0)
        y_pert = y + current_delta * s_random

        # ----------------------------------------------------
        # 1. L-Net optimization stage: Fix P-Net
        # ----------------------------------------------------
        self.l_model.train()
        self.p_net.eval()

        self.l_optimizer.zero_grad()

        # Martingale Loss
        with torch.no_grad():
            a = self.p_net(z, y_pert)
            # [B,obs_dim] -> [B,1,obs_dim]
            s_next = self.env.v_next(y_pert, a).unsqueeze(1)
            noise = triangular((B, 16, y.shape[1]), device=device, dtype=y.dtype)
            noise_scale = torch.as_tensor(self.env.noise, device=device, dtype=y.dtype).view(1, 1, -1)
            noise = noise * noise_scale
            # [B,16,obs_dim]
            s_next_random = s_next + noise
            s_next_random = s_next_random.detach() 

        # l value
        l = self.l_model(y_pert).view(-1)
        l_next = self.l_model(
            # [B*16,obs_dim]
            s_next_random.reshape(-1, s_next_random.size(-1))
        ).view(B, s_next_random.size(1))

        # [B,16] -> [B]
        exp_l_next = torch.mean(l_next, dim=1)
        violations_mean_l = (exp_l_next >= l).float().mean()

        # Martingale loss
        if self.gamma_decrease < 1.0:
            dec_loss = martingale_loss(self.gamma_decrease * l, exp_l_next, eps=0.0)
        else:
            dec_loss = martingale_loss(l, exp_l_next, eps=float(self.eps))
        
        loss_l = dec_loss * 1000

        # L-Net Lipschitz loss
        y_ = y.detach().clone().requires_grad_(True)
        l_out = self.l_model(y_).sum()
        l_grads = torch.autograd.grad(l_out, [y_], create_graph=True, retain_graph=True)
        grad_concat_l = torch.cat([g.view(g.size(0), -1) for g in l_grads], dim=1)
        local_lip_l = grad_concat_l.norm(p=2, dim=1)
        lip_loss_l = torch.relu(local_lip_l - self.l_lip).mean()
        loss_l = loss_l + lip_coeff * lip_loss_l

        # L-Net region loss
        if float(self.reach_prob) < 1.0:
            s_zero = torch.zeros(self.env.observation_space.shape[0], device=device)
            s_zero_batch = s_zero.unsqueeze(0)
            l_at_zero = self.l_model(s_zero_batch).view(-1)
            region_loss_zero = torch.sum(torch.maximum(torch.abs(l_at_zero), torch.tensor(0.03, device=device)))

            l_at_init = self.l_model(init_samples).view(-1)
            l_at_unsafe = self.l_model(unsafe_samples).view(-1)

            max_at_init = torch.max(l_at_init)
            min_at_unsafe = torch.min(l_at_unsafe)
            target_val = 1.0 / max(1e-6, (1.0 - float(self.reach_prob)))
            region_loss_unsafe = torch.maximum(torch.tensor(0.0, device=device), torch.tensor(target_val, device=device) - min_at_unsafe)
            region_loss_init = torch.maximum(torch.tensor(0.0, device=device), max_at_init - torch.tensor(1.0, device=device))
            
            region_loss = region_loss_zero + region_loss_unsafe + region_loss_init
            loss_l = loss_l + region_loss
        else:
            region_loss = torch.tensor(0.0, device=device)

        # L-Net optimization
        loss_l.backward()
        torch.nn.utils.clip_grad_norm_(self.l_model.parameters(), clip_grad)
        self.l_optimizer.step()

        # ----------------------------------------------------
        # 2. Indicator record
        # ----------------------------------------------------

        metrics = {
            "loss_l": loss_l.item(),
            "dec_loss_l": dec_loss.item(),
            "train_violations_l": violations_mean_l.item(),
            "lip_loss_l": lip_loss_l.item(),
            "region_loss": region_loss.item(),
        }
        return metrics
    

    def train_step_p(self, z, y, lip_coeff, current_delta, clip_grad=1.0):
        device = self.device
        B = y.shape[0]


        rng = torch.Generator(device=self.device)
        rng.manual_seed(19)
        s_random = torch.rand(y.shape, generator=rng, device=device) - 0.5
        current_delta = torch.tensor(current_delta, dtype=y.dtype, device=y.device).unsqueeze(0)
        y_pert = y + current_delta * s_random
        
        # ----------------------------------------------------
        #  1. P-Net optimization stage: Fix L-Net
        # ----------------------------------------------------
        self.l_model.eval()
        self.p_net.train()

        self.p_optimizer.zero_grad()
        
        # l value
        with torch.no_grad():
            l_p = self.l_model(y_pert).view(-1)
        
        a_p = self.p_net(z, y_pert)
        s_next_p = self.env.v_next(y_pert, a_p).unsqueeze(1)
        noise_p = triangular((B, 128, y.shape[1]), device=device, dtype=y.dtype)
        noise_scale = torch.as_tensor(self.env.noise, device=device, dtype=y.dtype).view(1, 1, -1)
        noise_p = noise_p * noise_scale
        s_next_random_p = s_next_p + noise_p
        
        # l_next
        with torch.no_grad():
            l_next_p = self.l_model(
                s_next_random_p.reshape(-1, s_next_random_p.size(-1))
            ).view(B, s_next_random_p.size(1))
        
        # l_next_p
        l_next_p_train = self.l_model(
            s_next_random_p.reshape(-1, s_next_random_p.size(-1))
        ).view(B, s_next_random_p.size(1))

        exp_l_next_p = torch.mean(l_next_p_train, dim=1)
        violations_mean_p = (exp_l_next_p >= l_p).float().mean()
        
        # Martingale loss 
        if self.gamma_decrease < 1.0:
            dec_loss_p = martingale_loss(self.gamma_decrease * l_p.detach(), exp_l_next_p, eps=0.0)
        else:
            dec_loss_p = martingale_loss(l_p.detach(), exp_l_next_p, eps=float(self.eps))

        loss_p = dec_loss_p * 10

        # P-Net Lipschitz loss
        gen_out = self.p_net.gen_net(z, y[:,0].unsqueeze(1)).detach().clone().requires_grad_(True)
        gen_out_flat = gen_out.view(gen_out.size(0), -1)
        state = self.p_net.state_net(gen_out_flat)
        y_col1 = y[:,1].unsqueeze(1).detach().clone().requires_grad_(True)
        controller_input = torch.cat([state, y_col1], dim=1)
        acc = self.p_net.controller_net(controller_input)
        acc = acc.sum()

        controller_grad = torch.autograd.grad(
            acc,
            controller_input,
            create_graph=False,
            allow_unused=True
        )[0]

        local_lip_p = controller_grad.view(controller_grad.size(0), -1).norm(p=2, dim=1)
        lip_loss_p = torch.relu(local_lip_p - self.p_lip).mean()
        loss_p = loss_p + lip_coeff * lip_loss_p

        with torch.no_grad():
            acc_labels = self.net(z, y_pert)
        mse_loss_p = F.mse_loss(a_p, acc_labels.view_as(a_p))
        loss_p = loss_p + mse_loss_p * 10

        # P-Net optimization
        loss_p.backward()
        torch.nn.utils.clip_grad_norm_(self.p_net.controller_net.parameters(), clip_grad)
        self.p_optimizer.step()

        # ----------------------------------------------------
        # 2. Indicator record
        # ----------------------------------------------------

        metrics = {
            "loss_p": loss_p.item(),
            "dec_loss_p": dec_loss_p.item(),
            "train_violations_p": violations_mean_p.item(),
            "lip_loss_p": lip_loss_p.item(),
            "mse_loss_p": mse_loss_p.item()
        }
        return metrics
    
    def train_step_joint(self, z, y, lip_coeff, current_delta, clip_grad=1.0):
        """
        Unified joint training:
        - Uses a single loss_total
        - Only 1 backward()
        - L-Net updated by martingale + L-Lip
        - P-Net updated by P-Lip + MSE
        - gen_net & state_net are frozen
        """
        device = self.device
        B = y.shape[0]

        # -------------------------
        # Step 0: Generate perturbed states
        # -------------------------
        rng = torch.Generator(device=device)
        rng.manual_seed(19)
        s_random = torch.rand(y.shape, generator=rng, device=device) - 0.5
        current_delta = torch.tensor(current_delta, dtype=y.dtype, device=device).unsqueeze(0)
        y_pert = y + current_delta * s_random

        # -------------------------
        # Zero Grad
        # -------------------------
        self.l_optimizer.zero_grad()
        self.p_optimizer.zero_grad()

        # =====================================================
        # 1. L-Net Forward Pass
        # =====================================================
        l_val = self.l_model(y_pert.float()).view(-1)  # <-- Force float32

        # =====================================================
        # 2. P-Net Forward Pass -> action
        # =====================================================
        a_p = self.p_net(z.float(), y_pert.float())  # <-- Force float32

        # Next state
        s_next = self.env.v_next(y_pert.float(), a_p).unsqueeze(1)  # <-- Force float32

        # =====================================================
        # 3. Martingale l_next -- Crucial detach!
        # =====================================================
        noise = triangular((B, 16, y.shape[1]), device=device, dtype=y.dtype)
        noise_scale = torch.as_tensor(self.env.noise, device=device).view(1, 1, -1)
        s_next_random = s_next + noise * noise_scale

        # Block the P-Net -> s_next -> L-next gradient flow
        s_next_for_l = s_next_random.detach().reshape(-1, y.shape[1])
        l_next = self.l_model(s_next_for_l.float()).view(B, -1)
        exp_l_next = l_next.mean(dim=1)

        # Martingale loss
        if self.gamma_decrease < 1.0:
            dec_loss = martingale_loss(self.gamma_decrease * l_val, exp_l_next, eps=0.0)
        else:
            dec_loss = martingale_loss(l_val, exp_l_next, eps=float(self.eps))

        martingale_loss_val = dec_loss * 50.0

        # =====================================================
        # 4. L-Net Lipschitz
        # =====================================================
        y_lip = y.detach().clone().float().requires_grad_(True)  # <-- Force float32
        l_out = self.l_model(y_lip).sum()
        l_grad = torch.autograd.grad(l_out, y_lip, create_graph=True)[0]
        l_lip_norm = l_grad.view(B, -1).norm(2, 1)
        lip_loss_l = torch.relu(l_lip_norm - self.l_lip).mean()

        # =====================================================
        # 5. P-Net Lipschitz on controller input
        # =====================================================
        # Manually construct controller input x = [state_net(img), v]
        d = y[:, 0].unsqueeze(1)
        img = self.p_net.gen_net(z.float(), d.float())  # <-- Force float32
        img_flat = img.view(img.size(0), -1)

        state = self.p_net.state_net(img_flat)  # requires_grad=False

        v = y[:, 1].unsqueeze(1).float()  # <-- Force float32

        # Controller input x
        x = torch.cat([state, v], dim=1).detach().clone().requires_grad_(True)

        phi = self.p_net.controller_net(x).sum()
        p_grad = torch.autograd.grad(phi, x, create_graph=True)[0]

        local_lip_p = p_grad.view(B, -1).norm(2, 1)
        lip_loss_p = torch.relu(local_lip_p - self.p_lip).mean()

        # =====================================================
        # 6. MSE Supervised Loss (using old model as Teacher Net)
        # =====================================================
        with torch.no_grad():
            acc_labels = self.net(z.float(), y_pert.float())

        mse_loss = F.mse_loss(a_p, acc_labels.view_as(a_p))

        # =====================================================
        # 7. Unified Loss (single total loss)
        # =====================================================
        loss_total = (
            martingale_loss_val +
            lip_coeff * lip_loss_l +
            lip_coeff * lip_loss_p +
            10.0 * mse_loss
        )

        # Single backward pass
        loss_total.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.l_model.parameters(), clip_grad)
        torch.nn.utils.clip_grad_norm_(self.p_net.controller_net.parameters(), clip_grad)

        # Update parameters
        self.l_optimizer.step()
        self.p_optimizer.step()

        return {
            "loss_total": loss_total.item(),
            "martingale": martingale_loss_val.item(),
            "dec_loss": dec_loss.item(),
            "lip_loss_l": lip_loss_l.item(),
            "lip_loss_p": lip_loss_p.item(),
            "mse_loss": mse_loss.item()
        }


    def create_bounded_module(self, model):
        """
        Create a BoundedModule from a model.
        """
        for name, param in model.named_parameters():
            # Freeze parameters; do not participate in training
            param.requires_grad = False

        state_input = torch.randn(1, self.env.observation_space.shape[0]).to(self.device)
        # Wrap the original model using the neural network verification tool
        lirpa_model = BoundedModule(model, state_input, device=self.device)

        return lirpa_model
    
    
    def train_epoch(
        self,
        train_ds,
        current_delta,
        lip=0.01,
        batch_size=256,
        shuffle=True,
        num_epochs=10,      
        train_fn='both'
    ):
        """
        train_ds: numpy array, [N, obs_dim], each row is a y sample
        current_delta: Perturbation magnitude
        lip: Lipschitz regularization coefficient
        batch_size: Size of each batch
        shuffle: Whether to shuffle data before each epoch
        num_epochs: Number of training epochs
        train_fn: Specify training function 'l', 'p', 'both'
        """
        N = train_ds.shape[0]

        # Convert numpy to torch
        all_y = torch.tensor(train_ds, dtype=torch.float32, device=self.device)

        epoch_metrics_list = []

        for epoch in range(num_epochs):
            if shuffle:
                indices = np.random.permutation(N)
                all_y = all_y[indices]

            batch_metrics = []
            num_batches = (N + batch_size - 1) // batch_size

            # with tqdm(total=num_batches, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                y_batch = all_y[start:end]

                # Generate random z in [-1, 1], batch size matches y_batch
                z_batch = (torch.rand(y_batch.size(0), 4, device=self.device) * 2.0 - 1.0).float()

                # Select training function
                if train_fn == 'l':
                    metrics = self.train_step_l(z_batch, y_batch, lip_coeff=lip, current_delta=current_delta)
                elif train_fn == 'p':
                    metrics = self.train_step_p(z_batch, y_batch, lip_coeff=lip, current_delta=current_delta)
                elif train_fn == 'both':
                    metrics = self.train_step_joint(z_batch, y_batch, lip_coeff=lip, current_delta=current_delta)
                else:  # Default to both
                    metrics_l = self.train_step_l(z_batch, y_batch, lip_coeff=lip, current_delta=current_delta)
                    metrics_p = self.train_step_p(z_batch, y_batch, lip_coeff=lip, current_delta=current_delta)
                    metrics = {**metrics_l, **metrics_p}

                batch_metrics.append(metrics)

            # Aggregate epoch metrics
            epoch_metrics = {k: np.mean([bm[k] for bm in batch_metrics]) for k in batch_metrics[0].keys()}
            epoch_metrics_list.append(epoch_metrics)

        return epoch_metrics_list