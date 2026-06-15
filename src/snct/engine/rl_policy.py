"""현 엔진(MVP) — DQN 기반 RL 적재 정책.
Gymnasium 환경(StowageEnv)에서 학습된 DQN 에이전트를 사용하거나,
체크포인트가 없는 경우 Greedy 폴백 기반 행동을 수행한다."""
import os
import numpy as np
import torch
import torch.nn as nn
from snct.engine.base import StowageStrategy
from snct.common.schema import YardState, CandidatePlan, Assignment


class DQNetwork(nn.Module):
    """Simple DQN network for slot selection."""

    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RLStrategy(StowageStrategy):
    name = "rl"

    def __init__(self, checkpoint: str | None = None, n_slots: int = 9):
        self.checkpoint = checkpoint
        self.n_slots = n_slots
        self.obs_dim = n_slots * 4 + 3
        self.model: DQNetwork | None = None

        if checkpoint and os.path.exists(checkpoint):
            self.model = DQNetwork(self.obs_dim, n_slots)
            self.model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
            self.model.eval()

    def _to_obs(self, yard: YardState, current_idx: int,
                occupied: dict[int, str], slot_weights: dict[int, float]) -> np.ndarray:
        """YardState → observation vector."""
        obs = np.zeros(self.obs_dim, dtype=np.float32)

        n = min(len(yard.slots), self.n_slots)
        for i in range(n):
            slot = yard.slots[i]
            base = i * 4
            obs[base + 0] = 1.0 if i in occupied else 0.0
            obs[base + 1] = slot_weights.get(i, 0.0) / 30.0
            obs[base + 2] = 1.0 if slot.dg_allowed else 0.0
            obs[base + 3] = 1.0 if slot.reefer_capable else 0.0

        if current_idx < len(yard.queue):
            c = yard.queue[current_idx]
            base = self.n_slots * 4
            obs[base + 0] = c.weight_ton / 30.0
            obs[base + 1] = 1.0 if c.dg else 0.0
            obs[base + 2] = 1.0 if c.reefer else 0.0

        return obs

    def _select_action(self, obs: np.ndarray, occupied: dict[int, str],
                       yard: YardState, current_idx: int) -> int:
        """Select best valid action using DQN or greedy fallback."""
        container = yard.queue[current_idx]
        n = min(len(yard.slots), self.n_slots)

        if self.model is not None:
            # Use DQN
            with torch.no_grad():
                q_values = self.model(torch.FloatTensor(obs)).numpy()

            # Mask invalid actions
            for i in range(n):
                if i in occupied:
                    q_values[i] = -float("inf")
                slot = yard.slots[i]
                if container.dg and not slot.dg_allowed:
                    q_values[i] = -float("inf")
                if container.reefer and not slot.reefer_capable:
                    q_values[i] = -float("inf")

            return int(np.argmax(q_values[:n]))
        else:
            # Greedy fallback: pick best valid slot by score
            best_idx, best_score = -1, -float("inf")
            for i in range(n):
                if i in occupied:
                    continue
                slot = yard.slots[i]
                if container.dg and not slot.dg_allowed:
                    continue
                if container.reefer and not slot.reefer_capable:
                    continue
                if container.weight_ton > slot.max_stack_weight:
                    continue
                score = (10 - slot.tier) * (container.weight_ton / 5.0)
                if score > best_score:
                    best_score = score
                    best_idx = i
            return best_idx if best_idx >= 0 else 0

    def _to_plan(self, assignments_raw: list[tuple[str, int]],
                 yard: YardState) -> CandidatePlan:
        """(container_id, slot_index) list → CandidatePlan."""
        assignments = []
        for cid, slot_idx in assignments_raw:
            if slot_idx < len(yard.slots):
                slot = yard.slots[slot_idx]
                assignments.append(
                    Assignment(container_id=cid, bay=slot.bay, row=slot.row, tier=slot.tier)
                )
        return CandidatePlan(
            assignments=assignments,
            engine="rl",
            objective={"assigned": len(assignments), "unassigned": len(yard.queue) - len(assignments)},
        )

    def plan(self, yard: YardState) -> CandidatePlan:
        """Run RL inference pipeline: observe → act → repeat for each container."""
        occupied: dict[int, str] = {}
        slot_weights: dict[int, float] = {}
        raw_assignments: list[tuple[str, int]] = []

        for idx, container in enumerate(yard.queue):
            obs = self._to_obs(yard, idx, occupied, slot_weights)
            action = self._select_action(obs, occupied, yard, idx)

            if 0 <= action < len(yard.slots):
                occupied[action] = container.id
                slot_weights[action] = slot_weights.get(action, 0.0) + container.weight_ton
                raw_assignments.append((container.id, action))

        return self._to_plan(raw_assignments, yard)
