"""Problem generator for LiquidWorld instances.

Generates solvable liquid-mixing planning problems with configurable:
  - Source purity: pure (single liquid) or mixture (multiple liquids per source)
  - Target count and complexity: number of targets × liquids per target
  - Source/target volumes

All generated problems are validated for solvability via volume checks,
concentration feasibility checks, and LP feasibility checks.
"""
import random
import itertools
import json
import copy
import os
from importlib import import_module
from typing import List, Dict
from liquid_world import LiquidWorldDomain, LiquidWorldState


LIQUID_NAMES = [f"L{i}" for i in range(1, 11)]
VESSEL_NAMES = [f"V{i}" for i in range(1, 50)]
INF_CAPACITY = 999.0
SOURCE_VOLUME = 10.0


def _round(x: float, precision: int = 2) -> float:
    return round(x, precision)


def _random_percentages(n: int, total: int = 100, step: int = 5) -> List[int]:
    """Generate n integer percentages (multiples of step) summing to total, each >= step."""
    if n * step > total:
        raise ValueError(f"Cannot generate {n} percentages >= {step} summing to {total}")
    pcts = [step] * n
    remaining = (total - step * n) // step
    for _ in range(remaining):
        pcts[random.randint(0, n - 1)] += step
    return pcts


def _check_lp_feasibility(
    source_containers: Dict[str, Dict[str, float]],
    goal_vols: Dict[str, float],
    target_name: str,
) -> bool:
    """Verify the target can be formed with non-negative pours from sources.

    Solves an LP feasibility problem where each variable x_i is volume poured
    from source i. Constraints enforce exact liquid volumes in the target for
    all liquids present in either the goal or sources (missing goal liquids
    have target volume 0), with bounds 0 <= x_i <= source_total_i.
    """
    sources = []
    for vessel, vols in source_containers.items():
        total = sum(vols.values())
        if total > 1e-9:
            sources.append((vessel, vols, total))

    if not sources:
        raise ValueError(f"Unsolvable: {target_name} has no available source vessels")

    try:
        linprog = import_module("scipy.optimize").linprog
    except Exception as exc:
        raise RuntimeError(
            "scipy is required for LP solvability checks. "
            "Install scipy to generate LiquidWorld datasets."
        ) from exc

    all_liquids = set(goal_vols.keys())
    for _, vols, _ in sources:
        all_liquids.update(vols.keys())
    all_liquids = sorted(all_liquids)

    a_eq = []
    b_eq = []
    for liquid in all_liquids:
        row = []
        for _, vols, total in sources:
            row.append(vols.get(liquid, 0.0) / total)
        a_eq.append(row)
        b_eq.append(goal_vols.get(liquid, 0.0))

    c = [1.0] * len(sources)
    bounds = [(0.0, total) for _, _, total in sources]

    result = linprog(c=c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success:
        raise ValueError(
            f"Unsolvable: {target_name} goal is not in conic hull of sources "
            f"(LP infeasible: {result.message})"
        )
    return True


def _check_joint_lp_feasibility(
    source_containers: Dict[str, Dict[str, float]],
    goal_containers: Dict[str, Dict[str, float]],
) -> bool:
    """Verify all targets are simultaneously satisfiable from shared sources.

    Unlike per-target LP, this accounts for source volume being shared:
    each source's total pours across ALL targets must not exceed its volume.

    Variables: x_{ij} = volume poured from source i into target j.
    Constraints:
      - For each (target j, liquid k): sum_i x_{ij} * conc_{ik} = goal_{jk}
      - For each source i: sum_j x_{ij} <= source_total_i
      - x_{ij} >= 0
    """
    sources = []
    for vessel, vols in source_containers.items():
        total = sum(vols.values())
        if total > 1e-9:
            sources.append((vessel, vols, total))

    if not sources:
        raise ValueError("Unsolvable: no available source vessels for joint LP")

    targets = list(goal_containers.items())
    n_src = len(sources)
    n_tgt = len(targets)

    try:
        linprog = import_module("scipy.optimize").linprog
    except Exception as exc:
        raise RuntimeError(
            "scipy is required for LP solvability checks."
        ) from exc

    all_liquids = set()
    for _, vols, _ in sources:
        all_liquids.update(vols.keys())
    for _, goal_vols in targets:
        all_liquids.update(goal_vols.keys())
    all_liquids = sorted(all_liquids)

    n_vars = n_src * n_tgt
    c = [1.0] * n_vars

    a_eq = []
    b_eq = []
    for j, (_, goal_vols) in enumerate(targets):
        for liq in all_liquids:
            row = [0.0] * n_vars
            for i, (_, vols, total) in enumerate(sources):
                row[i * n_tgt + j] = vols.get(liq, 0.0) / total
            a_eq.append(row)
            b_eq.append(goal_vols.get(liq, 0.0))

    a_ub = []
    b_ub = []
    for i, (_, _, total) in enumerate(sources):
        row = [0.0] * n_vars
        for j in range(n_tgt):
            row[i * n_tgt + j] = 1.0
        a_ub.append(row)
        b_ub.append(total)

    bounds = [(0.0, None) for _ in range(n_vars)]

    result = linprog(
        c=c, A_eq=a_eq, b_eq=b_eq, A_ub=a_ub, b_ub=b_ub,
        bounds=bounds, method="highs",
    )
    if not result.success:
        raise ValueError(
            f"Unsolvable: joint LP across {n_tgt} targets infeasible "
            f"({result.message})"
        )
    return True


def _validate_solvability(containers: Dict, goal_containers: Dict) -> bool:
    """Check that a problem is solvable: both volume and concentration feasibility.

    Volume check: total available volume of each liquid >= required.
    Concentration check: goal concentration of each liquid in each target
    must not exceed the max concentration of that liquid in any source.
    (You can never concentrate a liquid above its highest source concentration
    because pouring preserves ratios.)
    """
    available: Dict[str, float] = {}
    for vols in containers.values():
        for liquid, vol in vols.items():
            available[liquid] = available.get(liquid, 0.0) + vol

    required: Dict[str, float] = {}
    for vols in goal_containers.values():
        for liquid, vol in vols.items():
            required[liquid] = required.get(liquid, 0.0) + vol

    for liquid, need in required.items():
        have = available.get(liquid, 0.0)
        if need > have + 1e-9:
            raise ValueError(
                f"Unsolvable: {liquid} needs {need} L but only {have} L available"
            )

    # Concentration feasibility: for each liquid, the goal concentration in any
    # target cannot exceed the max source concentration of that liquid.
    max_conc: Dict[str, float] = {}
    for vols in containers.values():
        total = sum(vols.values())
        if total < 1e-9:
            continue
        for liquid, vol in vols.items():
            conc = vol / total
            max_conc[liquid] = max(max_conc.get(liquid, 0.0), conc)

    for target, goal_vols in goal_containers.items():
        goal_total = sum(goal_vols.values())
        if goal_total < 1e-9:
            continue
        for liquid, vol in goal_vols.items():
            goal_conc = vol / goal_total
            source_max = max_conc.get(liquid, 0.0)
            if goal_conc > source_max + 1e-9:
                raise ValueError(
                    f"Unsolvable: {target} needs {liquid} at "
                    f"{goal_conc*100:.1f}% but max source concentration is "
                    f"{source_max*100:.1f}%"
                )

    # LP feasibility check: verify targets admit non-negative solutions.
    source_containers = {
        vessel: vols
        for vessel, vols in containers.items()
        if vessel not in goal_containers and sum(vols.values()) > 1e-9
    }
    if len(goal_containers) <= 1:
        for target, goal_vols in goal_containers.items():
            _check_lp_feasibility(source_containers, goal_vols, target)
    else:
        _check_joint_lp_feasibility(source_containers, goal_containers)
    return True


def _create_mixture_sources(liquids: List[str], all_mixture: bool = False) -> List[Dict[str, float]]:
    """Create source vessel contents, some or all containing mixtures.

    Each source has exactly SOURCE_VOLUME (10L). Mixture sources have a primary
    liquid (50-80%) and 1-2 secondary liquids from the same liquid pool.

    Args:
        liquids: liquid type names available for mixing
        all_mixture: if True, every source is a mixture; if False, at least one stays pure
    """
    n = len(liquids)
    sources = []

    if all_mixture:
        mixed_indices = set(range(n))
    else:
        n_mixed = random.randint(1, max(1, n - 1))
        mixed_indices = set(random.sample(range(n), n_mixed))

    for i in range(n):
        if i not in mixed_indices:
            sources.append({liquids[i]: SOURCE_VOLUME})
        else:
            primary = liquids[i]
            primary_pct = random.choice([50, 55, 60, 65, 70, 75, 80])
            others = [j for j in range(n) if j != i]
            n_secondary = random.randint(1, min(2, len(others)))
            secondary_indices = random.sample(others, n_secondary)

            remaining_pct = 100 - primary_pct
            if n_secondary == 1:
                secondary_pcts = [remaining_pct]
            else:
                secondary_pcts = _random_percentages(n_secondary, total=remaining_pct)

            content = {primary: _round(SOURCE_VOLUME * primary_pct / 100)}
            for k, si in enumerate(secondary_indices):
                content[liquids[si]] = _round(SOURCE_VOLUME * secondary_pcts[k] / 100)

            total = sum(content.values())
            diff = _round(SOURCE_VOLUME - total)
            if abs(diff) > 0.001:
                content[primary] = _round(content[primary] + diff)

            sources.append(content)

    return sources


def _create_sources_with_config(
    source_config: List[int],
    liquids: List[str],
    liquid_groups: List[List[str]] = None,
    step: int = 5,
) -> List[Dict[str, float]]:
    """Create all-mixture sources following explicit per-source component counts.

    Args:
        source_config: Number of liquid components for each source vessel.
            Example: [3, 3, 3] means 3 sources each containing 3 liquids.
        liquids: Global liquid pool to sample from.
        liquid_groups: Optional disjoint liquid groups (e.g., per-target groups).
            If provided, each source is sampled from exactly one group.
        step: Percentage granularity for secondary splits.
    """
    if not source_config:
        raise ValueError("source_config must not be empty")

    n_liquids = len(liquids)
    if n_liquids == 0:
        raise ValueError("liquids must not be empty")

    for n_comp in source_config:
        if n_comp < 1:
            raise ValueError("Each source in source_config must have at least 1 component")
        if n_comp > n_liquids:
            raise ValueError(
                f"Invalid source_config entry {n_comp}: exceeds liquid pool size {n_liquids}"
            )

    if liquid_groups is None:
        liquid_groups = [list(liquids)]
    if not liquid_groups:
        raise ValueError("liquid_groups must not be empty")

    for group in liquid_groups:
        if not group:
            raise ValueError("Each liquid group must not be empty")

    # Retry random assignments until all liquids are covered at least once.
    assignments = None
    for _ in range(200):
        candidate = []
        covered = set()
        group_usage = [0] * len(liquid_groups)
        for n_comp in source_config:
            valid_group_ids = [
                idx for idx, group in enumerate(liquid_groups) if n_comp <= len(group)
            ]
            if not valid_group_ids:
                raise ValueError(
                    f"Cannot place source with {n_comp} components into available liquid groups"
                )
            group_id = random.choice(valid_group_ids)
            group = liquid_groups[group_id]
            picked_liquids = sorted(random.sample(group, n_comp))
            candidate.append(picked_liquids)
            covered.update(picked_liquids)
            group_usage[group_id] += 1

        all_groups_covered = all(
            all(liq in covered for liq in group) for group in liquid_groups
        )
        all_groups_satisfiable = all(
            set(group).issubset(
                set().union(*(set(sl) for sl in candidate if set(sl).issubset(set(group))))
                or set()
            )
            for group in liquid_groups
        )
        if all_groups_covered and all_groups_satisfiable:
            assignments = candidate
            break
    if assignments is None:
        raise ValueError(
            "Failed to assign source_config while covering all liquids. "
            "Increase overlap budget or retry with a different seed."
        )

    sources: List[Dict[str, float]] = []
    for source_liquids in assignments:
        n_comp = len(source_liquids)

        if n_comp == 1:
            sources.append({source_liquids[0]: SOURCE_VOLUME})
            continue

        primary_idx = random.randrange(n_comp)
        primary = source_liquids[primary_idx]
        primary_pct = random.choice([40, 45, 50, 55, 60, 65, 70])
        remaining_pct = 100 - primary_pct
        secondary_liquids = [liq for liq in source_liquids if liq != primary]

        if len(secondary_liquids) == 1:
            secondary_pcts = [remaining_pct]
        else:
            secondary_pcts = _random_percentages(
                len(secondary_liquids),
                total=remaining_pct,
                step=step,
            )

        content = {primary: _round(SOURCE_VOLUME * primary_pct / 100)}
        for liq, pct in zip(secondary_liquids, secondary_pcts):
            content[liq] = _round(SOURCE_VOLUME * pct / 100)

        total = sum(content.values())
        diff = _round(SOURCE_VOLUME - total)
        if abs(diff) > 0.001:
            content[primary] = _round(content[primary] + diff)
        sources.append(content)

    return sources


def _generate_problem(components_per_target: List[int],
                      initial_state_type: str,
                      difficulty: str = None,
                      source_config: List[int] = None,
                      n_sources: int = None,
                      source_volume=None,
                      target_volume=None) -> Dict:
    """Core generator: create a solvable LiquidWorld problem.

    Args:
        components_per_target: number of liquid types per target, e.g. [3] or [2, 4]
        initial_state_type: "pure" | "partial_mixture" | "all_mixture"
        difficulty: difficulty label string
        source_config: Optional all-mixture source shape, e.g. [2,3,3]
        n_sources: Total source vessels (for pure mode). Extra sources beyond
            the needed liquids become distractors. Defaults to n_liquids.
        source_volume: Volume per source. float (uniform) or list (per-source).
            Defaults to SOURCE_VOLUME (10L).
        target_volume: Volume per target. float (uniform), list (per-target),
            or None (random 2-5L per target).

    Layout:
      - Source vessels: each with SOURCE_VOLUME (pure or mixture depending on level)
      - Target vessels: empty workspace vessels for the goal
      For multi-target problems with enough sources (n_sources >= n_liquids),
      targets sample liquids from a shared pool (overlapping allowed) and
      sources are placed into overlapping liquid groups.  When sources are
      fewer than total distinct target liquids (shared pool), a deferred
      assignment finds feasible liquid subsets after source creation.
      A joint LP validates that all targets are simultaneously satisfiable.
    """
    n_targets = len(components_per_target)
    n_liquids = sum(components_per_target)
    if source_config is not None:
        _n_sources = len(source_config)
    elif n_sources is not None:
        _n_sources = n_sources
    else:
        _n_sources = n_liquids

    # Determine liquid pool and target assignments
    _shared_pool = _n_sources < n_liquids

    if n_targets == 1:
        all_liquids_count = max(_n_sources, n_liquids)
        liquids = [f"L{i}" for i in range(1, all_liquids_count + 1)]
        indices = list(range(all_liquids_count))
        random.shuffle(indices)
        assignments = [sorted(indices[:components_per_target[0]])]
    elif not _shared_pool:
        all_liquids_count = _n_sources
        liquids = [f"L{i}" for i in range(1, all_liquids_count + 1)]
        assignments = []
        for n_comp in components_per_target:
            if n_comp > all_liquids_count:
                raise ValueError(
                    f"Target needs {n_comp} components but only "
                    f"{all_liquids_count} liquids in pool"
                )
            selected = sorted(random.sample(range(all_liquids_count), n_comp))
            assignments.append(selected)
    else:
        all_liquids_count = _n_sources
        liquids = [f"L{i}" for i in range(1, all_liquids_count + 1)]
        assignments = None  # deferred until sources are created

    # Create source vessels
    containers = {}
    capacities = {}

    _src_vols = source_volume if source_volume is not None else [SOURCE_VOLUME] * _n_sources

    if initial_state_type == "pure":
        for i in range(_n_sources):
            v = VESSEL_NAMES[i]
            containers[v] = {liquids[i]: _src_vols[i]}
            capacities[v] = INF_CAPACITY

    elif initial_state_type == "partial_mixture":
        src_liquids = liquids[:_n_sources] if _n_sources < len(liquids) else liquids
        sources = _create_mixture_sources(src_liquids, all_mixture=False)
        for i in range(len(sources)):
            v = VESSEL_NAMES[i]
            containers[v] = sources[i]
            capacities[v] = INF_CAPACITY

    elif initial_state_type == "all_mixture":
        if source_config is not None:
            if n_targets == 1:
                liquid_groups = [[liquids[i] for i in assignments[0]]]
            elif not _shared_pool:
                liquid_groups = [[liquids[i] for i in asgn] for asgn in assignments]
            else:
                liquid_groups = None
            configured_sources = _create_sources_with_config(
                source_config,
                liquids,
                liquid_groups=liquid_groups,
            )
            for i, src in enumerate(configured_sources):
                v = VESSEL_NAMES[i]
                if _src_vols[i] != SOURCE_VOLUME:
                    scale = _src_vols[i] / SOURCE_VOLUME
                    containers[v] = {liq: _round(vol * scale) for liq, vol in src.items()}
                else:
                    containers[v] = src
                capacities[v] = INF_CAPACITY
        else:
            if n_targets == 1:
                group_liquids = [liquids[i] for i in assignments[0]]
            else:
                group_liquids = liquids[:all_liquids_count]
            group_sources = _create_mixture_sources(group_liquids, all_mixture=True)
            for i, gs in enumerate(group_sources):
                v = VESSEL_NAMES[i]
                containers[v] = gs
                capacities[v] = INF_CAPACITY

    # Deferred assignment for shared pool: pick feasible subsets based on actual sources
    if _shared_pool and assignments is None:
        source_vessels = VESSEL_NAMES[:_n_sources]
        feasible_cache = {}
        for n_comp in set(components_per_target):
            feasible = []
            for subset in itertools.combinations(range(all_liquids_count), n_comp):
                subset_liq = set(liquids[i] for i in subset)
                candidates = [
                    sv for sv in source_vessels
                    if set(containers[sv].keys()).issubset(subset_liq)
                ]
                covered = set()
                for sv in candidates:
                    covered.update(containers[sv].keys())
                if subset_liq.issubset(covered):
                    feasible.append(sorted(subset))
            feasible_cache[n_comp] = feasible

        assignments = []
        for n_comp in components_per_target:
            if not feasible_cache[n_comp]:
                raise ValueError(
                    f"No feasible {n_comp}-component subset for given sources"
                )
            assignments.append(random.choice(feasible_cache[n_comp]))

    # Create empty target vessels
    targets = []
    for t in range(n_targets):
        tv = VESSEL_NAMES[_n_sources + t]
        containers[tv] = {}
        capacities[tv] = INF_CAPACITY
        targets.append(tv)

    # Resolve target volumes (list, one per target; None = random)
    _tgt_vols = target_volume

    # Generate goal for each target
    goal_containers = {}
    goal_capacities = {}
    source_vessels = VESSEL_NAMES[:_n_sources]

    for t_idx, target in enumerate(targets):
        assigned = assignments[t_idx]
        assigned_liquids = [liquids[li_idx] for li_idx in assigned]
        total_volume = _tgt_vols[t_idx] if _tgt_vols else float(random.choice([2, 3, 4, 5]))

        if source_config is None:
            pcts = _random_percentages(len(assigned))
            goal_vols = {}
            for i, li_idx in enumerate(assigned):
                goal_vols[liquids[li_idx]] = _round(total_volume * pcts[i] / 100)

            diff = _round(total_volume - sum(goal_vols.values()))
            if abs(diff) > 0.001:
                first_key = liquids[assigned[0]]
                goal_vols[first_key] = _round(goal_vols[first_key] + diff)
        else:
            # Build goals from random convex combinations of feasible sources
            # to guarantee LP-feasible target compositions.
            candidate_sources = []
            assigned_set = set(assigned_liquids)
            for src_v in source_vessels:
                src_liquids = set(containers[src_v].keys())
                if src_liquids.issubset(assigned_set):
                    candidate_sources.append(src_v)

            if not candidate_sources:
                raise ValueError(
                    f"No valid sources for target {target} with assigned liquids {assigned_liquids}"
                )

            goal_vols = None
            for _ in range(200):
                raw_weights = [random.uniform(0.1, 1.0) for _ in candidate_sources]
                total_weight = sum(raw_weights)
                pours = [total_volume * w / total_weight for w in raw_weights]

                current = {liq: 0.0 for liq in assigned_liquids}
                for src_v, pour in zip(candidate_sources, pours):
                    src = containers[src_v]
                    src_total = sum(src.values())
                    for liq in assigned_liquids:
                        frac = src.get(liq, 0.0) / src_total
                        current[liq] += pour * frac

                if not all(v > 1e-6 for v in current.values()):
                    continue

                # Keep exact synthesized volumes to preserve LP feasibility in
                # high-dimensional targets (rounding can make overdetermined
                # systems infeasible).
                if all(v >= 0 for v in current.values()):
                    goal_vols = current
                    break

            if goal_vols is None:
                raise ValueError(
                    f"Failed to synthesize feasible goal for target {target} "
                    f"with assigned liquids {assigned_liquids}"
                )

        goal_containers[target] = goal_vols
        goal_capacities[target] = INF_CAPACITY

    # Round-trip goals through display format so internal targets match exactly
    # what models see in `goal_state_str` (prevents rounding inconsistency).
    tmp_goal_state = LiquidWorldState(goal_containers, goal_capacities)
    goal_state_str = tmp_goal_state.to_goal_str()
    parsed_goal_state = LiquidWorldDomain.parse_goal_description(goal_state_str)
    goal_containers = parsed_goal_state.containers
    goal_capacities = parsed_goal_state.capacities

    _validate_solvability(containers, goal_containers)

    state = LiquidWorldState(containers, capacities)
    goal_state = LiquidWorldState(goal_containers, goal_capacities)

    result = {
        "initial_state_str": str(state),
        "goal_state_str": goal_state.to_goal_str(),
        "containers": copy.deepcopy(containers),
        "capacities": dict(capacities),
        "goal_containers": goal_containers,
        "n_liquids": all_liquids_count,
        "n_sources": _n_sources,
        "n_targets": n_targets,
    }
    if difficulty is not None:
        result["difficulty"] = difficulty
    return result


def generate_problem(
    seed: int,
    state_type: str = "pure",
    difficulty: str = None,
    n_sources: int = None,
    components_per_target: List[int] = None,
    source_config: List[int] = None,
    source_volume=None,
    target_volume=None,
    max_attempts: int = 1,
) -> Dict:
    """Unified generator for LiquidWorld problems.

    Args:
        seed: Random seed.
        state_type: "pure" | "partial_mixture" | "all_mixture".
        difficulty: Optional label string stored in the output.
        n_sources: Total source vessels (pure mode distractors). Defaults to n_liquids.
        components_per_target: Component counts per target, e.g. [3] or [2,4].
        source_config: Per-source component counts for all_mixture mode.
        source_volume: float (uniform) or list (per-source). Defaults to 10L.
        target_volume: float (uniform), list (per-target), or None (random 2-5L).
        max_attempts: Retry count for mixture modes that may fail solvability.
    """
    for attempt in range(max_attempts):
        random.seed(seed + attempt)
        try:
            return _generate_problem(
                list(components_per_target), state_type, difficulty,
                source_config=source_config, n_sources=n_sources,
                source_volume=source_volume, target_volume=target_volume,
            )
        except ValueError:
            if attempt == max_attempts - 1:
                raise
            continue
    raise RuntimeError(f"Failed to generate solvable problem after {max_attempts} attempts (seed={seed})")


def generate_bench(config_file: str, output_file: str = None) -> List[Dict]:
    """Generate problems from a config file.

    The config is a JSON object with a global seed and a "problems" list.
    Each entry specifies source_components and components_per_target.

    Args:
        config_file: Path to JSON config.
        output_file: Output JSON path. Defaults to data/problems.json.
    """
    with open(config_file) as f:
        config = json.load(f)

    seed_base = config.get("seed", 42)
    problem_configs = config.get("problems", [])
    if not problem_configs:
        raise ValueError(f"No 'problems' list found in {config_file}")

    dataset = []
    for cfg in problem_configs:
        pid = cfg["id"]
        seed = seed_base + pid

        comps = cfg.get("components_per_target")
        if comps is None and "n_components" in cfg:
            comps = [cfg["n_components"]]

        src_comp = cfg.get("source_components")
        if src_comp is not None:
            n_sources = len(src_comp)
            if all(c == 1 for c in src_comp):
                state_type = "pure"
                source_config = None
            else:
                state_type = "all_mixture"
                source_config = src_comp
        else:
            n_sources = cfg.get("n_sources")
            state_type = cfg.get("state_type", "pure")
            source_config = cfg.get("source_config")

        p = generate_problem(
            seed=seed,
            state_type=state_type,
            difficulty=cfg.get("difficulty"),
            n_sources=n_sources,
            components_per_target=comps,
            source_config=source_config,
            source_volume=cfg.get("source_volume"),
            target_volume=cfg.get("target_volume"),
            max_attempts=cfg.get("max_attempts", 50),
        )
        p["id"] = pid
        dataset.append(p)

    if output_file is None:
        output_file = "data/problems.json"
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved {len(dataset)} problems to {output_file}")
    return dataset


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python liquid_world_generator.py <config_file> [output_file]")
        sys.exit(1)

    config_file = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else None

    dataset = generate_bench(config_file, output)
    for p in dataset:
        print(f"id={p.get('id', '?')} n_liquids={p['n_liquids']} n_targets={p['n_targets']}")
        print(f"  Init: {p['initial_state_str'][:120]}...")
        print(f"  Goal: {p['goal_state_str'][:120]}...")
        print()
