"""Run MEA planning on LiquidWorld problems.

Usage:
  python3 -m src.main --problem liquidword/data/level2.json --id 1
  python3 -m src.main --problem liquidword/data/level2.json --all
  python3 -m src.main --problem liquidword/data/level2.json --all --num-trials 3
  python3 -m src.main --problem liquidword/data/level2.json --id 1 --print-prompt
"""
import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from src.medaka.core import Medaka
from src.utils.config import Config


_SUPPORTED_PROVIDERS = {
    "openai", "anthropic", "azure_openai", "google",
    "dashscope", "openrouter", "together", "featherless",
}
_PROVIDER_MODEL_CONFIG_ATTRS = {
    "openai": "OPENAI_MODEL_NAME",
    "anthropic": "ANTHROPIC_MODEL_NAME",
    "azure_openai": "AZURE_OPENAI_DEPLOYMENT_NAME",
    "google": "GOOGLE_MODEL_NAME",
    "dashscope": "DASHSCOPE_MODEL_NAME",
    "openrouter": "OPENROUTER_MODEL_NAME",
    "together": "TOGETHER_MODEL_NAME",
    "featherless": "FEATHERLESS_MODEL_NAME",
}


def _get_model_display(provider):
    attr = _PROVIDER_MODEL_CONFIG_ATTRS.get(provider)
    return getattr(Config, attr) if attr else provider


def _apply_model_override(provider, model):
    attr = _PROVIDER_MODEL_CONFIG_ATTRS.get(provider)
    if attr and model:
        setattr(Config, attr, model)


def _mean_std(values):
    if not values:
        return 0.0, 0.0
    mean_v = sum(values) / len(values)
    var = sum((v - mean_v) ** 2 for v in values) / len(values)
    return mean_v, math.sqrt(var)


def _fmt_cost(mean_value, std_value):
    return f"{mean_value:.1f} +/- {std_value:.1f}"


def main():
    parser = argparse.ArgumentParser(description="Run MEA planning on LiquidWorld problems")
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--problem", type=str, required=True)
    parser.add_argument("--id", type=int, action="append", help="Problem ID(s)")
    parser.add_argument("--all", action="store_true", help="Run all problems in the dataset")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--model", type=str,
                        help="Override the configured model/deployment name for the selected provider")
    parser.add_argument("--output", type=str)
    parser.add_argument("--print-prompt", action="store_true",
                        help="Print the prompt and exit (no API calls)")
    parser.add_argument("--num-trials", type=int, default=1,
                        help="Number of independent trials per problem (default: 1)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of trials to run in parallel (default: 1 = sequential)")
    parser.add_argument("--reasoning-effort", type=str, default=None,
                        choices=["none", "low", "medium", "high", "xhigh"],
                        help="Reasoning effort for OpenAI models")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Override max output tokens")
    parser.add_argument("--enable-thinking", action="store_true", default=None,
                        help="Force enable thinking/reasoning mode")
    parser.add_argument("--no-thinking", dest="enable_thinking", action="store_false",
                        help="Force disable thinking/reasoning mode")
    args = parser.parse_args()

    if args.provider not in _SUPPORTED_PROVIDERS:
        parser.error(
            f"Unsupported --provider '{args.provider}'. "
            f"Supported: {', '.join(sorted(_SUPPORTED_PROVIDERS))}"
        )
        return

    if args.num_trials < 1:
        parser.error("--num-trials must be >= 1")
        return

    _apply_model_override(args.provider, args.model)
    model_display = _get_model_display(args.provider)
    reasoning_display = args.reasoning_effort or "default"
    thinking_display = {True: "on", False: "off", None: "auto"}[args.enable_thinking]
    print(
        f"Provider: {args.provider}, Model: {model_display}, "
        f"Trials: {args.num_trials}, "
        f"Reasoning: {reasoning_display}, Thinking: {thinking_display}"
    )

    with open(args.problem) as f:
        problems = json.load(f)

    if args.all and args.id:
        parser.error("--all cannot be combined with --id")
        return

    if args.all:
        targets = problems
    elif args.id:
        targets = [p for p in problems if p.get("id") in args.id]
    else:
        parser.error("Specify --id or --all")
        return

    # --- Print-prompt mode (no API calls) ---
    if args.print_prompt:
        solver = Medaka(args.provider,
                        reasoning_effort=args.reasoning_effort,
                        enable_thinking=args.enable_thinking)
        for p in targets:
            pid = p.get("id", "?")
            print(f"\n{'='*80}\nProblem {pid}\n{'='*80}")
            print(solver.get_prompt(p["initial_state_str"], p["goal_state_str"]))
        return

    # --- Solve ---
    results = []
    total_passed = 0

    for p in targets:
        pid = p.get("id", "?")
        init = p["initial_state_str"]
        goal = p["goal_state_str"]

        print(f"\n{'='*70}")
        print(f"Problem {pid}")
        print(f"  Init: {init[:100]}{'...' if len(init) > 100 else ''}")
        print(f"  Goal: {goal}")
        print(f"{'='*70}")

        trials = []
        problem_passed = 0
        jobs = list(range(1, args.num_trials + 1))

        def _run_one(trial_idx):
            solver = Medaka(args.provider,
                            reasoning_effort=args.reasoning_effort,
                            enable_thinking=args.enable_thinking)
            try:
                ok, steps, msg, out, usage = solver.solve(
                    init, goal,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            except Exception as e:
                ok, steps, msg, out, usage = False, [], str(e), "", {
                    "input": 0, "output": 0, "total": 0, "reasoning": 0,
                }
            return {"trial": trial_idx, "success": ok, "plan": steps,
                    "message": msg, "output": out, "usage": usage}

        n_workers = max(1, min(args.concurrency, len(jobs)))
        if n_workers > 1 and len(jobs) > 1:
            print(f"\n  [running {len(jobs)} trials with concurrency={n_workers}]")
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                completed = list(ex.map(_run_one, jobs))
        else:
            completed = [_run_one(j) for j in jobs]

        for c in completed:
            if args.num_trials > 1:
                print(f"\n  --- Trial {c['trial']}/{args.num_trials} ---")
            if c["success"]:
                problem_passed += 1
                total_passed += 1
            status = "PASS" if c["success"] else "FAIL"
            reasoning_tok = c["usage"].get("reasoning", 0)
            reasoning_info = f", reasoning={reasoning_tok}" if reasoning_tok else ""
            print(f"  {status}  ({len(c['plan'])} steps, "
                  f"{c['usage'].get('total', 0)} tok{reasoning_info})")
            for i, s in enumerate(c["plan"]):
                print(f"    {i+1}. {s}")
            trials.append(c)

        row = {"id": pid, "init": init, "goal": goal}
        if args.num_trials == 1:
            row.update({
                "success": trials[0]["success"],
                "plan": trials[0]["plan"],
                "message": trials[0]["message"],
                "output": trials[0]["output"],
                "usage": trials[0]["usage"],
            })
        else:
            row.update({
                "success_count": problem_passed,
                "num_trials": args.num_trials,
                "trials": trials,
            })
        results.append(row)

    # --- Summary ---
    n = len(targets)
    total_attempts = n * args.num_trials
    print(f"\n{'='*70}")
    print(f"SUMMARY: {total_passed}/{total_attempts} passed ({n} problems)")
    print(f"{'='*70}")
    for r in results:
        if args.num_trials == 1:
            status = "PASS" if r["success"] else "FAIL"
        else:
            status = f"{r['success_count']}/{args.num_trials}"
        print(f"  Problem {r['id']:<6} {status}")

    all_usage = []
    for r in results:
        if args.num_trials == 1:
            all_usage.append(r.get("usage", {}))
        else:
            for t in r.get("trials", []):
                all_usage.append(t.get("usage", {}))

    inputs = [int(u.get("input", 0) or 0) for u in all_usage]
    outputs = [int(u.get("output", 0) or 0) for u in all_usage]
    totals = [int(u.get("total", 0) or 0) for u in all_usage]
    im, ist = _mean_std(inputs)
    om, ost = _mean_std(outputs)
    tm, tst = _mean_std(totals)
    print(f"\nCOST (mean +/- std over {total_attempts} trials)")
    print(f"  Input:  {_fmt_cost(im, ist)}")
    print(f"  Output: {_fmt_cost(om, ost)}")
    print(f"  Total:  {_fmt_cost(tm, tst)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"results/run_{timestamp}.json"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "summary": {
                "passed": total_passed,
                "total_problems": n,
                "num_trials": args.num_trials,
                "total_attempts": total_attempts,
            },
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
