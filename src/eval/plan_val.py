"""LPDL plan validator — inspired by VAL (PDDL plan validator).

LPDL = Liquid Planning Description Language

Validates a plan step-by-step: checks preconditions, applies effects,
prints state transitions, and verifies the goal.

Usage:
  # Validate a specific problem with a given plan
  python3 -m src.lpdl.val --problem data/liquid_world_15.json --id 1 \
      --plan "Pour 1.65 L from V1 to V3; Pour 1.35 L from V2 to V3"

  # Validate a single problem from strings
  python3 -m src.lpdl.val \
      --init "V1 (capacity 999L): 10 L L1; V2 (capacity 999L): 10 L L2; V3 (capacity 999L): empty" \
      --goal "V3: 2 L, 50% L1, 50% L2" \
      --plan "Pour 1 L from V1 to V3; Pour 1 L from V2 to V3"
"""
import argparse
import json
import sys
from liquidword.domain.liquid_world import LiquidWorldState, LiquidWorldDomain


def validate(init_str: str, goal_str: str, plan_steps: list,
             tolerance: float = 0.05, verbose: bool = True) -> tuple:
    """VAL-style step-by-step plan validation.

    Returns:
        (success: bool, report: list of str)
    """
    report = []

    # --- Parse ---
    state = LiquidWorldDomain.parse_state_description(init_str)
    goal = LiquidWorldDomain.parse_goal_description(goal_str)

    report.append("=== Initial State ===")
    report.append(f"  {state}")
    report.append("")
    report.append("=== Goal ===")
    report.append(f"  {goal.to_goal_str()}")
    report.append("")
    report.append(f"=== Plan ({len(plan_steps)} steps) ===")

    sim = state.copy()

    for i, step_text in enumerate(plan_steps):
        report.append(f"--- Step {i+1}: {step_text} ---")

        parsed = LiquidWorldDomain.parse_action(step_text)
        if parsed is None:
            report.append(f"  FAIL: Cannot parse action '{step_text}'")
            _print_report(report, verbose)
            return False, report

        source, dest, volume = parsed

        # Precondition checks with specific error messages
        if source not in sim.containers:
            report.append(f"  FAIL: Source vessel '{source}' does not exist")
            _print_report(report, verbose)
            return False, report

        if dest not in sim.containers:
            report.append(f"  FAIL: Destination vessel '{dest}' does not exist")
            _print_report(report, verbose)
            return False, report

        if source == dest:
            report.append(f"  FAIL: Source and destination are the same vessel '{source}'")
            _print_report(report, verbose)
            return False, report

        if volume <= 0:
            report.append(f"  FAIL: Volume must be positive, got {volume}")
            _print_report(report, verbose)
            return False, report

        source_total = sim.total_volume(source)
        if source_total < volume - 1e-9:
            report.append(f"  FAIL: {source} has {source_total:.2f} L but need {volume:.2f} L")
            _print_report(report, verbose)
            return False, report

        remaining = sim.remaining_capacity(dest)
        if volume > remaining + 1e-9:
            report.append(f"  FAIL: {dest} has {remaining:.2f} L remaining capacity but pouring {volume:.2f} L")
            _print_report(report, verbose)
            return False, report

        # Preconditions OK — show details
        report.append(f"  Preconditions: OK")
        report.append(f"    {source}: {source_total:.2f} L available (pour {volume:.2f} L)")
        report.append(f"    {dest}: {remaining:.2f} L remaining capacity")

        # Apply
        sim.pour(source, dest, volume)
        report.append(f"  State after step {i+1}:")
        report.append(f"    {sim}")
        report.append("")

    # --- Goal check ---
    report.append("=== Goal Check ===")
    success = sim.is_goal_state(goal, tolerance)
    if success:
        report.append("  PASS: Goal reached!")
    else:
        report.append("  FAIL: Goal not reached")
        report.append(f"  Final:    {sim}")
        report.append(f"  Expected: {goal.to_goal_str()}")

    _print_report(report, verbose)
    return success, report


def _print_report(report: list, verbose: bool):
    if verbose:
        for line in report:
            print(line)


def main():
    parser = argparse.ArgumentParser(description="LPDL plan validator (VAL-style)")
    parser.add_argument("--problem", type=str, help="Path to JSON problem file from generator")
    parser.add_argument("--id", type=int, help="Problem ID to validate (use with --problem)")
    parser.add_argument("--init", type=str, help="Initial state string")
    parser.add_argument("--goal", type=str, help="Goal string")
    parser.add_argument("--plan", type=str, help="Plan steps separated by ';'")
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args()

    if args.init and args.goal and args.plan:
        plan_steps = [s.strip() for s in args.plan.split(";") if s.strip()]
        success, _ = validate(args.init, args.goal, plan_steps, args.tolerance)
        sys.exit(0 if success else 1)

    if not args.problem:
        parser.error("Provide --problem or (--init, --goal, --plan)")

    with open(args.problem) as f:
        problems = json.load(f)

    if args.id is not None:
        p = next((x for x in problems if x.get("id") == args.id), None)
        if p is None:
            print(f"Problem ID {args.id} not found")
            sys.exit(1)
        if not args.plan:
            parser.error("--plan is required when using --id")
        plan_steps = [s.strip() for s in args.plan.split(";") if s.strip()]
        success, _ = validate(
            p["initial_state_str"], p["goal_state_str"], plan_steps, args.tolerance
        )
        sys.exit(0 if success else 1)

    parser.error("Use --id N --plan '...' to validate a problem, or --init/--goal/--plan for strings")


if __name__ == "__main__":
    main()
