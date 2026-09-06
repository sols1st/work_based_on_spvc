import os
import time
import torch
import math
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


# Check if states are inside the interval
def v_contains(box, states):
    b_low = np.expand_dims(box.low, axis=0)
    b_high = np.expand_dims(box.high, axis=0)
    contains = np.logical_and(
        np.all(states >= b_low, axis=1), np.all(states <= b_high, axis=1)
    )
    return contains

# Check if states intersect with the interval
def v_intersect(box, lb, ub):
    b_low = np.expand_dims(box.low, axis=0)
    b_high = np.expand_dims(box.high, axis=0)
    contain_lb = np.logical_and(lb >= b_low, lb <= b_high)
    contain_ub = np.logical_and(ub >= b_low, ub <= b_high)
    # Must intersect in all dimensions
    contains_any = np.all(np.logical_or(contain_lb, contain_ub), axis=1)

    return contains_any

# Training Buffer
class TrainBuffer:
    def __init__(self, max_size=3_000_000):
        # List to store data blocks
        self.s = []
        # Maximum capacity
        self.max_size = max_size
        # Cached dataset (None if stale/needs reconstruction)
        self._cached_ds = None
    # Append a data block
    def append(self, s):
        if self.max_size is not None and len(self) > self.max_size:
            return
        self.s.append(s)
        self._cached_ds = None

    # Extend with a list of data blocks
    def extend(self, lst):
        for s in lst:
            self.append(s)
    # Return total number of samples
    def __len__(self):
        if len(self.s) == 0:
            return 0
        return sum([s.shape[0] for s in self.s])
    # Batch size of the first block (or input dimension)
    @property
    def in_dim(self):
        return len(self.s[0])
    # Rebuild the training dataset
    def as_tfds(self, batch_size=32):
        if self._cached_ds is not None:
            return self._cached_ds
        # Concatenate along the first dimension (axis=0)
        train_s = np.concatenate(self.s, axis=0)
        # Shuffle randomly
        rng = np.random.default_rng()
        train_s = rng.permutation(train_s)
        train_s = torch.tensor(train_s, dtype=torch.float32)

        # Build the Dataset
        train_dataset = TensorDataset(train_s)
        # Build the DataLoader
        train_ds = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2  # Similar to prefetch, improves loading efficiency
        )

        # Cache the DataLoader
        self._cached_ds = train_ds
        return train_ds
    

class VTVerifier:

    def __init__(
        self,
        vt_learner,
        env,
        l_ibp,
        batch_size,
        reach_prob,
        fail_check_fast
    ):
        # Value function learner
        self.learner = vt_learner
        self.l_ibp = l_ibp
        # Environment object
        self.env = env
        # Verification probability threshold
        self.reach_prob = np.float32(reach_prob)
        # Fast failure check
        self.fail_check_fast = fail_check_fast

        self.batch_size = batch_size
        self.refine_enabled = False

        # Space partitioning (grid size array, corresponding to dimensions)
        self.grid_size = env.space_split
        # Probability mass approximation partitioning
        self.pmass_n = 10 * np.ones(env.observation_space.shape[0]).astype(int)

        # Maximum stream size
        self.grid_stream_size = 1024 * 1024

        # Grid probability mass cache
        self._cached_pmass_grid = None
        # State grid cache
        self._cached_state_grid = None
        
        # Debug violations/Constraint violation status
        self._debug_violations = None
        # Hard constraint violation buffer
        self.hard_constraint_violation_buffer = None
        self.train_buffer = TrainBuffer()

        self.violation_buffer = []

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # Performance statistics
        self._perf_stats = {
            "apply": 0.0,
            "loop": 0.0,
        }

        # Vectorized computation
        self.v_get_grid_item = torch.vmap(
            self.get_grid_item, in_dims=(0,), out_dims=0
        )

    ################1. Discrete Grid Generation ######################
    # Pre-fill the training buffer
    def prefill_train_buffer(self):
        print("Creating pre-train buffer ... ", end="")
        # Generate discrete state grid
        state_grid, _, _ = self.get_unfiltered_grid(n = self.grid_size)
        self.train_buffer.append(state_grid)
        print(" [done]")

        # Return the grid step size for each dimension
        return (self.env.observation_space.high - self.env.observation_space.low) / self.grid_size
    
    # Calculate the coordinate for a specific grid index
    def get_grid_item(self, idx: torch.Tensor) -> torch.Tensor:
        low, high = self.env.observation_space.low, self.env.observation_space.high
        dims = low.shape[0]

        # Grid center points for each dimension (list)
        steps = (high - low) / self.grid_size
        target_points = [
            torch.linspace(
                low[i], 
                high[i], 
                self.grid_size[i] + 1
                )[:-1] + 0.5 * steps[i]
            for i in range(dims)
        ]

        # Linear idx -> Multi-dimensional idx (tensor)
        idxs = []
        tmp = idx
        for size in reversed(self.grid_size):
            idxs.append(tmp.remainder(size))
            tmp = tmp.div(size, rounding_mode='floor')
        idxs = torch.stack(list(reversed(idxs)))

        # Use gather for indexing, avoiding Python indexing
        coord = torch.stack([
            torch.gather(tp, 0, id.unsqueeze(0)).squeeze(0)
            for tp, id in zip(target_points, idxs)
        ], dim=0)

        return coord
    
    # Generate discrete grid for the environment state space
    def get_unfiltered_grid(self, n):
        dims = self.env.observation_space.low.shape[0]
        grid, steps = [], []
        for i in range(dims):
            samples, step = np.linspace(
                self.env.observation_space.low[i],
                self.env.observation_space.high[i],
                n[i],
                endpoint=False,
                retstep=True,
            )
            grid.append(samples)
            steps.append(step)
        grid = np.meshgrid(*grid)
        grid_lb = [x.flatten() for x in grid]
        grid_ub = [grid_lb[i] + steps[i] for i in range(dims)]
        grid_centers = [grid_lb[i] + steps[i] / 2 for i in range(dims)]

        grid_lb = np.stack(grid_lb, axis=1)
        grid_ub = np.stack(grid_ub, axis=1)
        grid_centers = np.stack(grid_centers, axis=1)

        return grid_centers, grid_lb, grid_ub
    
    # Noise perturbation added to the action
    def get_pmass_grid(self):
        if self._cached_pmass_grid is not None:
            return self._cached_pmass_grid
        # Perturbation magnitude of the action after denormalization
        dims = len(self.env.noise_bounds[0])
        grid, steps = [], []
        # Iterate over each noise dimension (noise needs self-implementation)
        for i in range(dims):
            samples, step = np.linspace(
                self.env.noise_bounds[0][i],
                self.env.noise_bounds[1][i],
                self.pmass_n[i],
                endpoint=False,
                retstep=True,
            )
            grid.append(samples)
            steps.append(step)
        # Generate grid combinations for all dimensions
        grid_lb = np.meshgrid(*grid)
        # Flatten lower bounds for each dimension
        grid_lb = [x.flatten() for x in grid_lb]
        # Get upper bounds for each dimension
        grid_ub = [grid_lb[i] + steps[i] for i in range(dims)]

        # Reshape to n^d x d form (lower bound + upper bound)
        batched_grid_lb = np.stack(grid_lb, axis=1)  
        batched_grid_ub = np.stack(grid_ub, axis=1)
        # Calculate integral probability for each grid cell
        pmass = self.env.integrate_noise(batched_grid_lb, batched_grid_ub)
        pmass = torch.tensor(pmass, dtype=torch.float32, device=self.device)
        batched_grid_lb = torch.tensor(batched_grid_lb, dtype=torch.float32, device=self.device)
        batched_grid_ub = torch.tensor(batched_grid_ub, dtype=torch.float32, device=self.device)
        self._cached_pmass_grid = (pmass, batched_grid_lb, batched_grid_ub)
        return pmass, batched_grid_lb, batched_grid_ub
    
    ################2. Barrier Function Bound Calculation ##########################
    # Initial set: compute maximum bound
    def compute_bound_init(self, n):
        _, grid_lb, grid_ub = self.get_unfiltered_grid(n)

        mask = np.zeros(grid_lb.shape[0], dtype=bool)
        for init_space in self.env.init_spaces:
            intersect = v_intersect(init_space, grid_lb, grid_ub)
            mask = np.logical_or(
                mask,
                intersect,
            )

        grid_lb = grid_lb[mask]
        grid_ub = grid_ub[mask]
        assert grid_ub.shape[0] > 0

        return self.compute_bounds_on_set(grid_lb, grid_ub)
    # Unsafe set: compute minimum bound
    def compute_bound_unsafe(self, n):
        _, grid_lb, grid_ub = self.get_unfiltered_grid(n)

        # Keep grid cells that intersect with the unsafe set
        mask = np.zeros(grid_lb.shape[0],dtype=bool)
        for unsafe_space in self.env.unsafe_spaces:
            intersect = v_intersect(unsafe_space, grid_lb, grid_ub)
            mask = np.logical_or(
                mask,
                intersect,
            )
        grid_lb = grid_lb[mask]
        grid_ub = grid_ub[mask]
        assert grid_ub.shape[0] > 0
        return self.compute_bounds_on_set(grid_lb, grid_ub)
    # Global domain bounds: used for normalization
    def compute_bound_domain(self, n):
        _, grid_lb, grid_ub = self.get_unfiltered_grid(n)

        assert grid_ub.shape[0] > 0
        return self.compute_bounds_on_set(grid_lb, grid_ub)
    
    def compute_bounds_on_set(self, grid_lb, grid_ub):
        # Initialize global lower and upper bounds
        global_min = torch.tensor(float('inf'))
        global_max = torch.tensor(float('-inf'))

        n_samples = grid_ub.shape[0]

        # Batch processing according to the defined batch_size
        for i in tqdm(range(int(np.ceil(n_samples / self.batch_size)))):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, n_samples)

            batch_lb = grid_lb[start:end]
            batch_ub = grid_ub[start:end]

            # Call interval propagation (IBP), returns lower and upper bounds
            batch_lb = torch.tensor(batch_lb, dtype=torch.float32, device=self.device)
            batch_ub = torch.tensor(batch_ub, dtype=torch.float32, device=self.device)
            # IBP
            lb, ub = self.l_ibp.compute_bounds((batch_lb,batch_ub), method="IBP")
            # lb, ub = self.l_ibp.compute_bounds((batch_lb,batch_ub), method="CROWN")

            # Update global lower and upper bounds
            global_min = torch.min(global_min, torch.min(lb))
            global_max = torch.max(global_max, torch.max(ub))

        return float(global_min), float(global_max)
    
    # Compute the expected upper bound (using discrete grid centers s). This function needs modification!!!
    def compute_expected_l(self, s, a, pmass, batched_grid_lb, batched_grid_ub):
        # Get the next deterministic state
        deterministic_s_next = self.env.v_next(s, a)
        batch_size = s.shape[0]
        ibp_size = batched_grid_lb.shape[0]
        # Dimension
        obs_dim = self.env.observation_space.shape[0]

        # Broadcasting happens here, that's why we don't do directly vmap (although it's probably possible somehow)
        deterministic_s_next = deterministic_s_next.reshape((batch_size, 1, obs_dim))
        batched_grid_lb = batched_grid_lb.reshape((1, ibp_size, obs_dim))
        batched_grid_ub = batched_grid_ub.reshape((1, ibp_size, obs_dim))

        # Add perturbation to the state (due to previous +/- setup!!!)
        batched_grid_lb = batched_grid_lb + deterministic_s_next
        batched_grid_ub = batched_grid_ub + deterministic_s_next

        # Flatten (N * obs_dim)
        batched_grid_lb = batched_grid_lb.reshape((-1, obs_dim))
        batched_grid_ub = batched_grid_ub.reshape((-1, obs_dim))
        # Get the output upper bound for each cell (B(S) should decrease)
        lb, ub = self.l_ibp.compute_bounds((batched_grid_lb,batched_grid_ub), method="IBP")
        # Discrete states * corresponding perturbation count
        ub = ub.reshape((batch_size, ibp_size))

        pmass = pmass.reshape((1, ibp_size))  # Boradcast to batch size
        # Get the weighted expectation of corresponding perturbations for each state
        exp_terms = pmass * ub
        # Sum over perturbations for each state to get the expected upper bound
        expected_value = torch.sum(exp_terms, axis=1)
        return expected_value
    
    # Compute the local Lipschitz constant upper bound
    def compute_local_lipschitz_batch(self, s):
        s = s.clone().detach().requires_grad_(True)
        l_out = self.learner.l_model(s)
        l_out_sum = l_out.sum()

        # Backward gradient (calculate ∂f/∂x)
        grads = torch.autograd.grad(l_out_sum, s, create_graph=False)[0]  # shape: (batch_size, 2)

        # Jacobian norm (L2 norm) for each sample
        lip_norms = grads.norm(p=2, dim=1)  # shape: (batch_size,)
        return lip_norms.detach()
    
    # Check if the expected Lyapunov function decrease condition is satisfied for a batch of states
    def _check_dec_batch(self, s, a, l_batch, K):
        # Get the perturbation grid
        pmass, batched_grid_lb, batched_grid_ub = self._cached_pmass_grid
        # Get the expected upper bound for each state
        e = self.compute_expected_l(
            s,
            # Current batch of states
            a,
            pmass,
            batched_grid_lb,
            batched_grid_ub,
        )
        # l_batch is the Lyapunov value for the current state
        # Soft constraint
        decrease = e + K - l_batch
        # Count violations
        violating_indices = decrease >= 0
        v = violating_indices.sum()
        # Hard constraint
        hard_violating_indices = e - l_batch >= 0
        hard_v = hard_violating_indices.sum()
        return v, violating_indices, hard_v, hard_violating_indices
    # Normalize
    def normalize(self, l, ub_init, domain_min):
        # domain_min is the global minimum, ub_init is the initial region maximum
        l = l - domain_min
        ub_init = ub_init - domain_min

        l = l / max(ub_init, 1e-6)
        return l
    
    def check_dec_cond(self, k_except_l):
        # Total number of grid points in the state space
        grid_total_size = math.prod(self.grid_size)

        loop_start_time = time.perf_counter()
        
        # Lyapunov upper bound for the initial region
        _, ub_init = self.compute_bound_init(self.grid_size)
        unsafe_min, _ = self.compute_bound_unsafe(self.grid_size)
        # Global domain lower bound
        domain_min, _ = self.compute_bound_domain(self.grid_size)

        info_dict = {}
        # Grid step calculation
        diff = (self.env.observation_space.high - self.env.observation_space.low) / self.grid_size
        delta = np.linalg.norm(diff/2, ord=2).item()
        K = k_except_l * delta
        info_dict["delta"] = delta
        info_dict["K"] = K
        print(f"delta={delta}")
        print(f"Checking GRID of size {self.grid_size}")
        # Generate and cache perturbation grid probability
        self.get_pmass_grid()  # cache pmass grid
        
        violations = 0
        hard_violations = 0
        # violation_buffer = []
        hard_violation_buffer = []

        # Control stream size
        grid_stream_size = min(grid_total_size, self.grid_stream_size)
        total_kernel_time = 0
        total_kernel_iters = 0
        pbar = tqdm(total=grid_total_size // grid_stream_size)
        # Iterate over the grid
        for i in range(0, grid_total_size, grid_stream_size):
            # Get grid items based on index
            idx = torch.arange(i, i + grid_stream_size)
            sub_grid = self.v_get_grid_item(idx)
            
            kernel_start = time.perf_counter()
            for start in range(0, sub_grid.shape[0], self.batch_size):
                end = min(start + self.batch_size, sub_grid.shape[0])
                # Current batch grid
                s_batch = sub_grid[start:end]
                s_batch = s_batch.to(self.device)
                z_batch = torch.zeros(s_batch.shape[0], 4, dtype=torch.float32, device=self.device)
                a_batch = self.learner.p_net(z_batch, s_batch)
                # TODO: later optimize this by filtering the entire stream first
                # Compute Lyapunov value for the state
                l_batch = self.learner.l_model(s_batch).flatten()
                # Compute local Lipschitz upper bound for the L function
                lips_l_batch = self.compute_local_lipschitz_batch(s_batch)
                # Normalize
                normalized_l_batch = self.normalize(l_batch, ub_init, domain_min)
                # Preliminary filtering: only need to check states below a threshold
                # less_than_p = normalized_l_batch - lips_l_batch*K < 1 / (1 - self.reach_prob)
                normalized_unsafe_min = self.normalize(unsafe_min, ub_init, domain_min) 
                less_than_p = (normalized_l_batch > 0.95 * normalized_unsafe_min) & (normalized_l_batch < normalized_unsafe_min)
                if (less_than_p.int().sum().item()) == 0:
                    violation_buffer_np = None
                    continue
                s_batch = s_batch[less_than_p]
                a_batch = a_batch[less_than_p]
                l_batch = l_batch[less_than_p]
                lips_l_batch = lips_l_batch[less_than_p]
                (
                    v,
                    violating_indices,
                    hard_v,
                    hard_violating_indices,
                ) = self._check_dec_batch(
                    s_batch,
                    a_batch,
                    l_batch,
                    lips_l_batch*K,
                )
                
                hard_violations += hard_v
                violations += v

                # Add counterexamples
                violation_buffer_np = None
                if self.refine_enabled and v > 0:
                    temp = s_batch[violating_indices].cpu().numpy()  # Move to CPU and convert to numpy
                    self.violation_buffer.append(temp)
                    violation_buffer_np = np.concatenate(self.violation_buffer, axis=0)
                if hard_v > 0:
                    hard_violation_buffer.append(s_batch[hard_violating_indices])
            pbar.update(1)
            if i > 0:
                total_kernel_time += time.perf_counter() - kernel_start
                total_kernel_iters += sub_grid.shape[0] // self.batch_size
                ints_per_sec = (
                    total_kernel_iters * self.batch_size / total_kernel_time / 1000
                )
                pbar.set_description(
                    f"kernel_t: {total_kernel_time*1e6/total_kernel_iters:0.2f}us/iter ({ints_per_sec:0.2f} Kints/s)"
                )
            if self.fail_check_fast and violations > 0:
                break
        pbar.close()
        print(f"violations={hard_violations}")
        # print(f"hard_violations={hard_violations}")

        loop_time = time.perf_counter() - loop_start_time

        # info_dict["dec_violations"] = f"{violations}/{grid_total_size}"
        info_dict["hard_violations"] = f"{hard_violations}/{grid_total_size}"
        # print(f"{violations}/{grid_total_size} violated decrease condition")
        print(f"{hard_violations}/{grid_total_size} violations")
        print(f"Train buffer len: {len(self.train_buffer)}")

        if loop_time > 60:
            print(f"Grid runtime={loop_time/60:0.0f} min")
        else:
            print(f"Grid runtime={loop_time:0.2f} s")

        flag = False
        # if hard_violations / len(self.train_buffer) <= 0.01:
        if hard_violations / len(self.train_buffer) <= 0.001:
            flag = True
        return flag, hard_violations, info_dict, violation_buffer_np