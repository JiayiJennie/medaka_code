"""LiquidWorld domain implementation for AoT+ experiments.

Based on liquid_mixing, with LiquidWorld format:
- State: Vi (capacity x L): y L Ln (pure) | y L, z% L1, w% L2 (mixture) | empty
- Goal: Vi: x L, y% L1, z% L2
- Action: Pour x L from Vi to Vj (volume-based, step 3)
"""
from typing import Dict, List, Tuple, Optional
import re
import copy


def _natural_key(s: str):
    """Sort key that handles embedded numbers: V2 < V10, L1 < L2 < L10."""
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', s)]


class LiquidWorldState:
    """Representation of a state in the LiquidWorld domain.

    Each vessel holds a homogeneous mixture of liquids. Internal format:
    containers: {vessel: {liquid: volume}}, capacities: {vessel: float}
    """

    def __init__(self, containers: Dict[str, Dict[str, float]], capacities: Dict[str, float]):
        self.containers = containers
        self.capacities = capacities

    def total_volume(self, vessel: str) -> float:
        return sum(self.containers.get(vessel, {}).values())

    def remaining_capacity(self, vessel: str) -> float:
        return self.capacities[vessel] - self.total_volume(vessel)

    def get_ratios(self, vessel: str) -> Dict[str, float]:
        total = self.total_volume(vessel)
        if total == 0:
            return {}
        return {liquid: vol / total for liquid, vol in self.containers[vessel].items() if vol > 0}

    def pour(self, source: str, destination: str, volume: float) -> bool:
        """Pour a volume of liquid from source to destination.

        Args:
            source: Vessel name
            destination: Vessel name
            volume: Volume in L to pour

        Returns:
            True if valid and performed, False otherwise
        """
        if source not in self.containers or destination not in self.containers:
            return False
        if source == destination:
            return False
        if volume <= 0:
            return False

        source_total = self.total_volume(source)
        if source_total < volume - 1e-9:
            return False
        if volume > self.remaining_capacity(destination) + 1e-9:
            return False

        fraction = volume / source_total if source_total > 0 else 0
        for liquid, vol in list(self.containers[source].items()):
            transfer = vol * fraction
            self.containers[source][liquid] -= transfer
            if liquid not in self.containers[destination]:
                self.containers[destination][liquid] = 0.0
            self.containers[destination][liquid] += transfer

        self._cleanup()
        return True

    def _cleanup(self):
        for vessel in self.containers:
            self.containers[vessel] = {
                liquid: vol for liquid, vol in self.containers[vessel].items()
                if vol > 1e-9
            }

    def __str__(self) -> str:
        """LiquidWorld format: pure liquid | mixture | empty."""
        parts = []
        for name in sorted(self.containers.keys(), key=_natural_key):
            cap = self.capacities[name]
            cap_str = str(int(cap)) if cap == int(cap) else f"{cap:.1f}"
            total = self.total_volume(name)
            if total < 1e-9:
                parts.append(f"{name} (capacity {cap_str}L): empty")
            else:
                ratios = self.get_ratios(name)
                if len(ratios) == 1:
                    liquid, vol = list(self.containers[name].items())[0]
                    vol_str = str(int(vol)) if vol == int(vol) else f"{vol:.2f}"
                    parts.append(f"{name} (capacity {cap_str}L): {vol_str} L {liquid}")
                else:
                    conc_parts = ", ".join(
                        f"{round(r * 100, 2)}% {li}" for li, r in sorted(ratios.items(), key=lambda x: _natural_key(x[0]))
                    )
                    total_str = str(int(total)) if total == int(total) else f"{total:.2f}"
                    parts.append(f"{name} (capacity {cap_str}L): {total_str} L, {conc_parts}")
        return "; ".join(parts)

    def is_goal_state(self, goal_state: 'LiquidWorldState', tolerance: float = 0.05) -> bool:
        """Check if this state satisfies the goal. Goal is a partial state (target vessels only)."""
        for vessel, target_vols in goal_state.containers.items():
            if vessel not in self.containers:
                return False
            for liquid, target_vol in target_vols.items():
                actual_vol = self.containers[vessel].get(liquid, 0.0)
                if abs(actual_vol - target_vol) > tolerance:
                    return False
            target_total = sum(target_vols.values())
            actual_total = self.total_volume(vessel)
            if abs(actual_total - target_total) > tolerance:
                return False
        return True

    def to_goal_str(self) -> str:
        """Goal format: Vi: x L, y% L1, z% L2 (no capacity)."""
        parts = []
        for vessel in sorted(self.containers.keys(), key=_natural_key):
            vols = self.containers[vessel]
            total = sum(vols.values())
            if total < 1e-9:
                continue
            sorted_items = sorted(vols.items(), key=lambda x: _natural_key(x[0]))
            raw_pcts = [round(v / total * 100, 2) for _, v in sorted_items]
            residual = round(100.0 - sum(raw_pcts), 2)
            if residual != 0.0:
                max_idx = max(range(len(raw_pcts)), key=lambda i: raw_pcts[i])
                raw_pcts[max_idx] = round(raw_pcts[max_idx] + residual, 2)
            conc_parts = ", ".join(
                f"{pct}% {li}" for (li, _), pct in zip(sorted_items, raw_pcts)
            )
            total_str = str(int(total)) if total == int(total) else f"{total:.2f}"
            parts.append(f"{vessel}: {total_str} L, {conc_parts}")
        return "; ".join(parts)

    def copy(self) -> 'LiquidWorldState':
        return LiquidWorldState(
            copy.deepcopy(self.containers),
            dict(self.capacities)
        )


class LiquidWorldDomain:
    """LiquidWorld planning domain."""

    @staticmethod
    def get_domain_description() -> str:
        return """
# LiquidWorld Planning Domain

## Description:
You have vessels with fixed capacities. Vessels hold homogeneous mixtures of liquids (L1, L2, L3, ...).
All liquids have the same density and blend when mixed. You pour a specified volume from one vessel to another.

## Rules:
1. Pour only from non-empty vessels.
2. Specify volume in L to pour (e.g., 0.5 L).
3. Poured liquid preserves the mixture ratio of the source.
   When pouring from a mixture, ALL liquid types transfer in proportion to their concentration.
4. Destination must have enough remaining capacity.
5. Any non-empty vessel can serve as a source, including a goal vessel that has been partially filled.
   A goal vessel only needs to match its target state at the END of the plan; intermediate contents are unconstrained.

## State and Goal Format:
- Pure liquid: Vi (capacity x L): y L Ln
- Mixture: Vi (capacity x L): y L, z% L1, w% L2
- Empty: Vi (capacity x L): empty
- Goal: Vi: x L, y% L1, z% L2

## Action Format:
- Pour x L from Vi to Vj
"""

    @staticmethod
    def parse_state_description(description: str) -> LiquidWorldState:
        """Parse LiquidWorld state format into state object."""
        containers: Dict[str, Dict[str, float]] = {}
        capacities: Dict[str, float] = {}

        # Split by ; for multiple vessels
        parts = [p.strip() for p in description.split(";") if p.strip()]
        for part in parts:
            # V1 (capacity 1L): 1 L L1  or  1 L, 60% L1, 40% L2  or  empty
            vessel_match = re.match(r"(\w+)\s*\(capacity\s*([\d.]+)\s*L?\)\s*:\s*(.+)", part, re.I)
            if not vessel_match:
                continue
            vessel = vessel_match.group(1)
            cap = float(vessel_match.group(2))
            contents = vessel_match.group(3).strip()

            capacities[vessel] = cap
            if contents.lower() == "empty":
                containers[vessel] = {}
            else:
                containers[vessel] = {}
                # Check for pure liquid: "1 L L1"
                pure_match = re.match(r"([\d.]+)\s*L\s+(L\d+)\s*$", contents, re.I)
                if pure_match:
                    vol, liquid = float(pure_match.group(1)), pure_match.group(2)
                    containers[vessel][liquid] = vol
                else:
                    # Mixture: "1 L, 60% L1, 40% L2"
                    mix_match = re.match(r"([\d.]+)\s*L\s*,\s*(.+)", contents, re.I)
                    if mix_match:
                        total = float(mix_match.group(1))
                        rest = mix_match.group(2)
                        for pct_match in re.finditer(r"([\d.]+)\s*%\s*(L\d+)", rest, re.I):
                            pct, li = float(pct_match.group(1)) / 100, pct_match.group(2)
                            containers[vessel][li] = total * pct

        return LiquidWorldState(containers, capacities)

    @staticmethod
    def parse_goal_description(description: str) -> LiquidWorldState:
        """Parse goal format: V3: 2 L, 50% L1, 50% L2. Returns partial state (goal)."""
        containers: Dict[str, Dict[str, float]] = {}
        capacities: Dict[str, float] = {}
        parts = [p.strip() for p in description.split(";") if p.strip()]
        for part in parts:
            goal_match = re.match(r"(\w+)\s*:\s*([\d.]+)\s*L\s*,\s*(.+)", part, re.I)
            if goal_match:
                vessel = goal_match.group(1)
                total = float(goal_match.group(2))
                rest = goal_match.group(3)
                vols = {}
                for pct_match in re.finditer(r"([\d.]+)\s*%\s*(L\d+)", rest, re.I):
                    pct, li = float(pct_match.group(1)) / 100, pct_match.group(2)
                    vols[li] = total * pct
                containers[vessel] = vols
                capacities[vessel] = total
        return LiquidWorldState(containers, capacities)

    @staticmethod
    def generate_example_problem() -> Tuple[str, str, List[str], List[List[str]]]:
        initial_state = (
            "V1 (capacity 1L): 1 L L1; "
            "V2 (capacity 1L): 1 L L2; "
            "V3 (capacity 2L): empty"
        )
        goal_state = (
            "V1: 0.5 L L1"
            "V2: 0.5 L L2"
            "V3: 1 L, 50% L1, 50% L2"
        )
        solution = [
            "Pour 0.5 L from V1 to V3",
            "Pour 0.5 L from V2 to V3",
        ]
        random_trajectory1 = [
            "Pour 0.5 L from V1 to V3",
            "Pour 1 L from V2 to V3",
        ]
        random_trajectory2 = [
            "Pour 4 L from V1 to V3",
        ]
        return initial_state, goal_state, solution, [random_trajectory1, random_trajectory2]

    @staticmethod
    def create_example_state() -> Tuple[LiquidWorldState, LiquidWorldState]:
        state = LiquidWorldState(
            containers={"V1": {"L1": 1.0}, "V2": {"L2": 1.0}, "V3": {}},
            capacities={"V1": 1.0, "V2": 1.0, "V3": 2.0}
        )
        goal_state = LiquidWorldState(
            containers={"V3": {"L1": 1.0, "L2": 1.0}},
            capacities={"V3": 2.0}
        )
        return state, goal_state

    @staticmethod
    def parse_action(action_text: str) -> Optional[Tuple[str, str, float]]:
        """Parse 'Pour x L from Vi to Vj' -> (source, dest, volume).

        Supports both decimal (0.232) and fraction (16/69) volume formats.
        """
        pattern = r"[Pp]our\s+([\d./]+)\s*L\s*(?:\([^)]*\)\s*)?from\s+(\w+)\s+to\s+(\w+)"
        match = re.search(pattern, action_text)
        if match:
            vol_str = match.group(1)
            if '/' in vol_str:
                num, den = vol_str.split('/', 1)
                volume = float(num) / float(den)
            else:
                volume = float(vol_str)
            source = match.group(2)
            dest = match.group(3)
            return source, dest, volume
        return None

    @staticmethod
    def validate_plan(
        state: LiquidWorldState,
        goal_state: LiquidWorldState,
        plan_steps: List[str],
        tolerance: float = 0.05
    ) -> Tuple[bool, str]:
        sim_state = state.copy()
        for i, step in enumerate(plan_steps):
            parsed = LiquidWorldDomain.parse_action(step)
            if parsed is None:
                return False, f"Step {i+1}: Could not parse action '{step}'"
            source, dest, volume = parsed
            if not sim_state.pour(source, dest, volume):
                return False, f"Step {i+1}: Invalid pour - '{step}' (state: {sim_state})"
        if sim_state.is_goal_state(goal_state, tolerance):
            return True, f"Plan valid! Final state: {sim_state}"
        return False, f"Goal not reached. Final state: {sim_state}"
