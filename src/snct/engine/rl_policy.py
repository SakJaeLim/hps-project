"""현 엔진(MVP) — PPO 기반 RL 적재 정책.
Gymnasium 환경(SingleBayStowageEnv)에서 학습된 PPO 에이전트(BL, SF, EF)를 사용하거나,
체크포인트가 없는 경우 Greedy 폴백 기반 행동을 수행한다."""
import os
import sys
import numpy as np
import torch
import gymnasium as gym
from snct.engine.base import StowageStrategy
from snct.common.schema import YardState, CandidatePlan, Assignment, Slot, Container

# PPO Constants matching the training notebook
MAX_ROWS = 10
N_TIERS = 10
N_POD = 6
MIN_WT = 10.0
MAX_WT = 20.0
MAX_COL_WT = 145.0
OBS_DIM = 79

POD_MAP = {
    "BUSAN": 1, "BUSAN(1)": 1,
    "SHANGHAI": 2, "SHANGHAI(2)": 2,
    "NINGBO": 3, "NINGBO(3)": 3,
    "SINGAPORE": 4, "SINGAPORE(4)": 4,
    "COLOMBO": 5, "COLOMBO(5)": 5,
    "ROTTERDAM": 6, "ROTTERDAM(6)": 6,
    "LAX": 6,  # Default fallback
}

# --- Apply NumPy and pickle patches for loading compatibility ---
# Set NumPy alias mapping
sys.modules['numpy._core'] = np.core
sys.modules['numpy.random._pcg64'] = np.random

import numpy.random._pickle as _pickle
from numpy.random import PCG64
_pickle.BitGenerators[PCG64] = PCG64

class RLStrategy(StowageStrategy):
    name = "rl"

    def __init__(self, model_type: str = "BL", checkpoint_dir: str | None = None):
        self.model_type = model_type
        self.model = None
        
        try:
            from stable_baselines3 import PPO
            # Determine base dir
            base_dir = os.environ.get("SNCT_BASE_DIR", r"c:\Users\lione\Desktop\aSSIST\19_Project\12_hps-project-main")
            if not checkpoint_dir:
                checkpoint_dir = os.path.join(base_dir, "data", "RL", "강화학습 결과 자료", "single_bay_6pod_ppo_v13_3way_ALL_models_seed42")
            
            # Explicitly target the .zip file to avoid directory permission/loading errors
            model_path = os.path.join(checkpoint_dir, f"single_bay_6pod_ppo_v13_3way_{model_type}_seed42.zip")
            
            if os.path.exists(model_path) and os.path.isfile(model_path):
                # Define spaces explicitly to override deserialization mismatch
                observation_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(OBS_DIM,), dtype=np.float32)
                action_space = gym.spaces.Discrete(MAX_ROWS)

                custom_objects = {
                    "observation_space": observation_space,
                    "action_space": action_space
                }
                
                self.model = PPO.load(model_path, device="cpu", custom_objects=custom_objects)
                print(f"[RLStrategy] Loaded PPO model type: {model_type} from {model_path}")
            else:
                print(f"[RLStrategy] Warning: Checkpoint path not found or is not a file: {model_path}. Running with greedy fallback.")
        except Exception as e:
            print(f"[RLStrategy] Error loading PPO model: {e}. Running with greedy fallback.")

    def _to_obs(self, wt_grid: np.ndarray, pod_grid: np.ndarray, stack_h: np.ndarray,
                ctns_wt: np.ndarray, ctns_pod: np.ndarray, step_idx: int,
                n_containers: int, n_valid: int) -> np.ndarray:
        """Create normalized 79-dimensional observation vector matching the notebook."""
        mid_row = (MAX_ROWS - 1) / 2.0 + 1e-9

        # 1. Stack features (MAX_ROWS * 6 = 60)
        stack_feats = np.zeros((MAX_ROWS, 6), dtype=np.float32)
        for r in range(MAX_ROWS):
            h = int(stack_h[r])
            if h == 0:
                stack_feats[r] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                continue
            fill = h / N_TIERS
            wt_norm = float(wt_grid[r, :h].sum()) / (MAX_WT * N_TIERS)
            mean_pod = float(pod_grid[r, :h].mean()) / N_POD
            top_pod = float(pod_grid[r, h - 1]) / N_POD
            
            ov = sum(1 for t in range(h - 1) if pod_grid[r, t] < pod_grid[r, t + 1])
            ov_ratio = ov / max(h - 1, 1)
            
            col_wt = float(wt_grid[r, :h].sum())
            col_wt_ratio = col_wt / MAX_COL_WT
            
            stack_feats[r] = [fill, wt_norm, mean_pod, top_pod, ov_ratio, col_wt_ratio]

        # 2. Current container features (1 + N_POD = 7)
        if step_idx < n_containers:
            w = float(ctns_wt[step_idx]) / MAX_WT
            pod = int(ctns_pod[step_idx])
            pod_oh = np.eye(N_POD, dtype=np.float32)[pod - 1]
            cur_feats = np.array([w, *pod_oh], dtype=np.float32)
        else:
            cur_feats = np.zeros(1 + N_POD, dtype=np.float32)

        # 3. Global features (6)
        total_wt = float(wt_grid.sum()) + 1e-9
        row_wts = wt_grid.sum(axis=1)
        row_wts_nonzero = row_wts[row_wts > 0]

        row_idx = np.arange(MAX_ROWS, dtype=np.float32)
        cog = float(np.dot(row_idx, row_wts)) / total_wt
        cog_norm = (cog - mid_row) / mid_row

        if len(row_wts_nonzero) >= 2:
            wt_cv = float(np.std(row_wts_nonzero)) / (float(np.mean(row_wts_nonzero)) + 1e-9)
        else:
            wt_cv = 0.0

        n_placed = step_idx
        total_ov = sum(
            sum(1 for t in range(int(stack_h[r]) - 1) if pod_grid[r, t] < pod_grid[r, t + 1])
            for r in range(MAX_ROWS)
        )

        global_feats = np.array([
            step_idx / n_containers,
            float(np.clip(cog_norm, -2.0, 2.0)),
            float(np.clip(wt_cv, 0.0, 2.0)),
            n_valid / max(n_placed, 1),
            total_ov / max(n_placed, 1),
            n_placed / (MAX_ROWS * N_TIERS),
        ], dtype=np.float32)

        # 4. Remaining POD distribution (N_POD = 6)
        rem = ctns_pod[step_idx:]
        if len(rem) > 0:
            counts = np.array([np.sum(rem == p) for p in range(1, N_POD + 1)], dtype=np.float32)
            rem_dist = counts / (counts.sum() + 1e-9)
        else:
            rem_dist = np.zeros(N_POD, dtype=np.float32)

        return np.concatenate([
            stack_feats.flatten(),
            cur_feats,
            global_feats,
            rem_dist,
        ]).astype(np.float32)

    def plan(self, yard: YardState) -> CandidatePlan:
        # Sort containers to match curriculum ordering (POD descending, weight descending)
        containers = list(yard.queue)
        def get_sort_key(c):
            pod_name = str(c.pod).upper()
            pod_idx = POD_MAP.get(pod_name, 6)
            return (-pod_idx, -c.weight_ton)
            
        sorted_indices = sorted(range(len(containers)), key=lambda i: get_sort_key(containers[i]))
        
        n_containers = len(containers)
        ctns_wt = np.array([c.weight_ton for c in containers], dtype=np.float32)[sorted_indices]
        ctns_pod = np.array([POD_MAP.get(str(c.pod).upper(), 6) for c in containers], dtype=np.int32)[sorted_indices]
        
        # Initialize board representation (10x10 grids)
        wt_grid = np.zeros((MAX_ROWS, N_TIERS), dtype=np.float32)
        pod_grid = np.zeros((MAX_ROWS, N_TIERS), dtype=np.int32)
        stack_h = np.zeros(MAX_ROWS, dtype=np.int32)
        
        # Output assignments: maps container index -> Slot
        assigned_slots = {}
        occupied_slots = []
        n_valid = 0
        
        for step_idx in range(n_containers):
            orig_idx = sorted_indices[step_idx]
            container = containers[orig_idx]
            
            # 1. Predict action using PPO if loaded
            ppo_action = -1
            if self.model is not None:
                obs = self._to_obs(wt_grid, pod_grid, stack_h, ctns_wt, ctns_pod, step_idx, n_containers, n_valid)
                action, _ = self.model.predict(obs, deterministic=True)
                ppo_action = int(action)
                
            # 2. Try the predicted action first
            chosen_slot = None
            if 0 <= ppo_action < MAX_ROWS:
                # Find the lowest empty slot in the chosen row
                target_row = ppo_action + 1
                row_slots = [s for s in yard.slots if s.row == target_row and s not in occupied_slots]
                if row_slots:
                    # Sort by tier ascending
                    row_slots = sorted(row_slots, key=lambda s: s.tier)
                    candidate = row_slots[0]
                    # Check validation constraints (Reefer, DG, Weight)
                    is_valid = True
                    if container.dg and not candidate.dg_allowed:
                        is_valid = False
                    if container.reefer and not candidate.reefer_capable:
                        is_valid = False
                    if wt_grid[ppo_action, :stack_h[ppo_action]].sum() + container.weight_ton > candidate.max_stack_weight:
                        is_valid = False
                        
                    if is_valid:
                        chosen_slot = candidate
                        n_valid += 1
                        
            # 3. Fallback to greedy search if PPO action is invalid or fails constraints
            if chosen_slot is None:
                best_slot, best_score = None, -float("inf")
                for s in yard.slots:
                    if s in occupied_slots:
                        continue
                    if container.dg and not s.dg_allowed:
                        continue
                    if container.reefer and not s.reefer_capable:
                        continue
                    # Check row weight capacity
                    row_idx_grid = s.row - 1
                    if row_idx_grid < MAX_ROWS:
                        current_row_wt = wt_grid[row_idx_grid, :stack_h[row_idx_grid]].sum()
                    else:
                        current_row_wt = 0.0
                        
                    if current_row_wt + container.weight_ton > s.max_stack_weight:
                        continue
                        
                    # Greedy scoring rule
                    score = (10 - s.tier) * (container.weight_ton / 5.0)
                    if score > best_score:
                        best_score = score
                        best_slot = s
                chosen_slot = best_slot
                
            # 4. Apply assignment if slot is found
            if chosen_slot is not None:
                assigned_slots[orig_idx] = chosen_slot
                occupied_slots.append(chosen_slot)
                
                # Update internal grid state for observations
                r_idx = chosen_slot.row - 1
                if 0 <= r_idx < MAX_ROWS:
                    t_idx = min(stack_h[r_idx], N_TIERS - 1)
                    wt_grid[r_idx, t_idx] = container.weight_ton
                    pod_grid[r_idx, t_idx] = POD_MAP.get(str(container.pod).upper(), 6)
                    stack_h[r_idx] += 1
            else:
                # No slot could be allocated (yard full or constraint violation)
                pass

        # Build final plan
        assignments = []
        for idx in range(n_containers):
            if idx in assigned_slots:
                s = assigned_slots[idx]
                assignments.append(
                    Assignment(
                        container_id=containers[idx].id,
                        bay=s.bay,
                        row=s.row,
                        tier=s.tier
                    )
                )
                
        return CandidatePlan(
            assignments=assignments,
            engine=f"rl_{self.model_type.lower()}",
            objective={
                "assigned": len(assignments),
                "unassigned": n_containers - len(assignments)
            }
        )
