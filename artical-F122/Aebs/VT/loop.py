import os
import time
import math
import sys
import argparse
import torch
import numpy as np
from Aebs.system.env import Aebs
from Aebs.VT.verify import VTVerifier
from Aebs.VT.train import VTLearner

class Loop:
    def __init__(
        self,
        learner,
        verifier,
        env,
        jitter_grid,
        soft_constraint
    ):
        # Environment definition
        self.env = env
        # Learner (Lyapunov function + policy function)
        self.learner = learner
        # Verifier
        self.verifier = verifier
        # Whether soft constraint is enabled
        self.soft_constraint = soft_constraint
        # Custom grid partition
        self.jitter_grid = jitter_grid

        # Initial discretization interval
        self.prefill_delta = 0

        # Iteration counter
        self.iter = 0
        # Record iteration information
        self.info = {}
    
    # Training function (Lyapunov function + policy)
    def train(self, model, num_epochs=10, violation_buffer = None):
        # 1. Prepare dataset based on the discretized samples from run()!!!
        #    (discretized center points + discretization interval)
        train_ds, grid_lb, grid_ub = self.verifier.get_unfiltered_grid(self.env.train_space_split)
        current_delta = (self.env.observation_space.high - self.env.observation_space.low) / self.env.train_space_split

        start_metrics = None
        batch_size = 2048
        lip_coeff = 0.001
        if model == 'p' and violation_buffer is not None:
            train_ds = violation_buffer
            # train_ds = np.concatenate([train_ds, violation_buffer])
        # Record training time
        start_time = time.time()

        epoch_metrics_list = self.learner.train_epoch(
            train_ds=train_ds,
            current_delta=current_delta,
            lip=lip_coeff,
            batch_size=batch_size,
            shuffle=True,
            num_epochs = num_epochs,
            train_fn = model
        )
        
        # Print epoch metrics
        for epoch_idx, metrics in enumerate(epoch_metrics_list, start=1):
            print(f"Epoch {epoch_idx}:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
            print("-" * 40)
        
        # Record buffer size, i.e., number of verification grid points
        self.info["ds_size"] = len(self.verifier.train_buffer)
        # 3. Print training information
        elapsed = time.time() - start_time
        if elapsed > 60:
            elapsed = f"{elapsed/60:0.1f} minutes"
        else:
            elapsed = f"{elapsed:0.1f} seconds"

    # Verification + training main loop
    def run(self, timeout):
        start_time = time.time()
        last_saved = time.time()
        # 1. Generate global discretized grid of the environment
        self.prefill_delta = self.verifier.prefill_train_buffer()
        max_reach_prob = 0
        actual_reach_prob = 0
        hard_violation_list = []  # NEW: record hard_violations
        prob_list = []
        prob = 0
        while True:
            runtime = time.time() - start_time
            # Exit on timeout
            if runtime > timeout:
                print("Timeout!")
                break  # changed to break so mean values can be computed afterward

            # NEW: iteration limit check
            if self.iter > 100:
                print("Iteration limit reached (100). Stop.")
                break

            print(f"\n#### Iteration {self.iter} ({runtime // 60:0.0f}:{runtime % 60:02.0f} elapsed) #####")

            # 2. Train Lyapunov function first
            self.train('l', 10)
            # self.train('both', num_epochs=10)

            # 3. Estimate Lipschitz constant
            k_except_l = float(1.2)
            self.info["K"] = k_except_l
            self.info["iter"] = self.iter
            self.info["runtime"] = runtime

            # 4. Verify the decrease condition
            sat, hard_violations, info, violation_buffer = self.verifier.check_dec_cond(k_except_l)
            for k, v in info.items():
                self.info[k] = v
            print("info=", str(self.info), flush=True)

            if isinstance(hard_violations, torch.Tensor):
                hard_violation_list.append(hard_violations.detach().cpu().numpy())
            else:
                hard_violation_list.append(hard_violations)

            # 5. If decrease condition satisfied, estimate probability bound
            if sat:
                print("Decrease condition fulfilled!")

                _, ub_init = self.verifier.compute_bound_init(self.jitter_grid)
                lb_unsafe, _ = self.verifier.compute_bound_unsafe(self.jitter_grid)
                domain_min, _ = self.verifier.compute_bound_domain(self.jitter_grid)
                print(f"Init   max = {ub_init:0.6g}")
                print(f"Unsafe min = {lb_unsafe:0.6g}")
                print(f"domain min = {domain_min:0.6g}")
                self.info["ub_init"] = ub_init
                self.info["lb_unsafe"] = lb_unsafe
                self.info["domain_min"] = domain_min

                bound_correct = True
                if lb_unsafe < ub_init:
                    bound_correct = False
                    print("RSM is lower at unsafe than in init. No probabilistic guarantees can be obtained.")
                    self.info["actual_reach_prob"] = "UNSAFE"
                else:
                    ub_init = ub_init - domain_min
                    lb_unsafe = lb_unsafe - domain_min
                    lb_unsafe = lb_unsafe / ub_init

                    actual_reach_prob = 1 - 1 / np.clip(lb_unsafe, 1e-9, None)
                    if actual_reach_prob > max_reach_prob:
                        # torch.save(self.learner.l_model.state_dict(), "./Aebs/l_model.pth")
                        # torch.save(self.learner.p_net.state_dict(), "./Aebs/p_net.pth")
                        max_reach_prob = actual_reach_prob
                        print("[SAVED]")
                    prob = max_reach_prob
                    print(f"Probability of reaching the target safely is at least {actual_reach_prob*100:0.3f}% (higher is better)")

                # if self.soft_constraint or actual_reach_prob >= self.verifier.reach_prob:
                #     break  # changed to break, for unified mean computation later
            
            prob_list.append(prob)
            sys.stdout.flush()
            self.train('p', 1)
            self.iter += 1
        
        # np.save("./exp_Carla/hard_violation_list_050.npy", np.array(hard_violation_list, dtype=object))
        # np.save("./exp_Carla/prob_list_050.npy", np.array(prob_list, dtype=float))
        return True

if __name__ == "__main__":


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize environment
    env = Aebs(0.05)

    # Initialize learner
    vt_learner = VTLearner(
        l_model_config=[2, 16, 8, 1],
        env=env,
        p_lip=2.0,
        l_lip=4.0,
        eps=0.1, 
        gamma_decrease=1.0,
        reach_prob=0.95,
        square_l_output=True,
        # l_model_path="./Aebs/Aebs_l_model.pth",
        # p_model_path="./Aebs/Aebs_p_net.pth"
    )

    # --------------------------------------------------------------------------------------
    # Initialize verifier
    l_ibp = vt_learner.create_bounded_module(vt_learner.l_model)
    vt_verifier = VTVerifier(
        vt_learner,
        env,
        l_ibp,
        batch_size=2048,
        # batch_size=1024,
        reach_prob=0.9,
        fail_check_fast=True,
    )
    # ----------------------------------------------------------------------------------------
    # Initialize loop controller
    loop = Loop(
        vt_learner,
        vt_verifier,
        env,
        jitter_grid=env.space_split,
        soft_constraint=False,
    )

    # Run main loop
    sat = loop.run(60 * 60)

    # Log results
    with open("info.log", "a") as f:
        f.write("sat=" + str(sat) + "\n")
        f.write("info=" + str(loop.info) + "\n\n\n")
