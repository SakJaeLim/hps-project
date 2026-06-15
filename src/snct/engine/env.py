"""L3 Gym 환경 래퍼 — 컨테이너 적재 최적화 RL 환경.
상태=야드/베이 점유+큐, 액션=슬롯 배정, 보상=-재취급 -제약위반 -중량불균형."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from snct.common.schema import YardState, Container, Slot


class StowageEnv(gym.Env):
    """Container Stowage Planning RL Environment.

    Observation: [slot_occupied(0/1), slot_weight_ratio, slot_dg, slot_reefer,
                   current_container_weight_ratio, current_dg, current_reefer]
    Action: slot index (Discrete)
    Reward: positive for valid placement, negative for constraint violations.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, n_slots: int = 9, n_containers: int = 6):
        super().__init__()
        self.n_slots = n_slots
        self.n_containers = n_containers

        # Action: pick a slot index
        self.action_space = spaces.Discrete(n_slots)

        # Observation: per-slot features (4) + current container features (3)
        obs_dim = n_slots * 4 + 3
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self.yard_state: YardState | None = None
        self.current_idx: int = 0
        self.occupied: dict[int, str] = {}  # slot_index -> container_id
        self.slot_weights: dict[int, float] = {}  # slot_index -> total weight

    def _generate_scenario(self) -> YardState:
        """Generate a random stowage scenario for training."""
        rng = np.random.default_rng()

        slots = []
        for i in range(self.n_slots):
            bay = i // 3 + 1
            row = i % 3 + 1
            tier = 1
            dg_allowed = rng.random() > 0.7
            reefer_capable = rng.random() > 0.7
            slots.append(
                Slot(
                    bay=bay, row=row, tier=tier,
                    max_stack_weight=30.0,
                    dg_allowed=dg_allowed,
                    reefer_capable=reefer_capable,
                )
            )

        containers = []
        pods = ["LAX", "ROTTERDAM", "SINGAPORE", "BUSAN", "SHANGHAI"]
        for i in range(self.n_containers):
            weight = float(rng.uniform(5, 28))
            is_dg = rng.random() > 0.8
            is_reefer = rng.random() > 0.8 and not is_dg
            ctype = "DG" if is_dg else ("RF" if is_reefer else "GP")
            containers.append(
                Container(
                    id=f"C-{i:03d}",
                    weight_ton=round(weight, 1),
                    size="40",
                    type=ctype,
                    pod=rng.choice(pods),
                    dg=is_dg,
                    reefer=is_reefer,
                    discharge_order=i,
                )
            )

        return YardState(slots=slots, queue=containers)

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(self.n_slots * 4 + 3, dtype=np.float32)

        for i, slot in enumerate(self.yard_state.slots):
            base = i * 4
            obs[base + 0] = 1.0 if i in self.occupied else 0.0
            obs[base + 1] = self.slot_weights.get(i, 0.0) / 30.0
            obs[base + 2] = 1.0 if slot.dg_allowed else 0.0
            obs[base + 3] = 1.0 if slot.reefer_capable else 0.0

        # Current container features
        if self.current_idx < len(self.yard_state.queue):
            c = self.yard_state.queue[self.current_idx]
            base = self.n_slots * 4
            obs[base + 0] = c.weight_ton / 30.0
            obs[base + 1] = 1.0 if c.dg else 0.0
            obs[base + 2] = 1.0 if c.reefer else 0.0

        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.yard_state = self._generate_scenario()
        self.current_idx = 0
        self.occupied = {}
        self.slot_weights = {}
        return self._get_obs(), {}

    def step(self, action: int):
        assert self.yard_state is not None
        container = self.yard_state.queue[self.current_idx]
        slot = self.yard_state.slots[action]

        reward = 0.0
        violated = False

        # Check if slot is already occupied
        if action in self.occupied:
            reward -= 10.0
            violated = True

        # Check DG constraint
        if container.dg and not slot.dg_allowed:
            reward -= 5.0
            violated = True

        # Check Reefer constraint
        if container.reefer and not slot.reefer_capable:
            reward -= 5.0
            violated = True

        # Check weight capacity
        current_weight = self.slot_weights.get(action, 0.0)
        if current_weight + container.weight_ton > slot.max_stack_weight:
            reward -= 3.0
            violated = True

        if not violated:
            # Valid placement
            reward += 2.0

            # Heavy-down bonus (lower tier = better for heavy)
            reward += (container.weight_ton / 30.0) * (1.0 / max(slot.tier, 1))

            # POD grouping bonus
            for idx, cid in self.occupied.items():
                other_slot = self.yard_state.slots[idx]
                for c in self.yard_state.queue:
                    if c.id == cid and c.pod == container.pod:
                        if other_slot.bay == slot.bay:
                            reward += 0.5

            self.occupied[action] = container.id
            self.slot_weights[action] = current_weight + container.weight_ton

        self.current_idx += 1
        done = self.current_idx >= len(self.yard_state.queue)
        truncated = False

        return self._get_obs(), reward, done, truncated, {"violated": violated}
