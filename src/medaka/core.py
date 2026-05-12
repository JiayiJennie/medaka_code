"""Core implementation of the Medaka planning methodology."""

import re
from typing import List, Optional, Tuple

from liquidword.domain.liquid_world import LiquidWorldDomain


class MedakaPromptBuilder:
    """Builder for MEA-style prompts with structured reasoning format."""

    STRUCTURED_FORMAT = """\
### MEA Planning Format

Think step by step to explore possible actions.
You MUST follow this format strictly for every planning step.

### Key Instructions:
1. Compare the current state with the goal state and list all differences (Diff)
2. Pick one difference and search for an operator that reduces it
3. Judge whether the operator is directly solvable
   (solvable = all preconditions met AND all parameters determined)
4. If not solvable, set "make the operator solvable" as a subgoal
   and repeat from step 2 for that subgoal until a solvable operator is found
5. Execute the solvable operator, update the state, and return to step 1 for remaining differences
6. If the result deviates from expectation, diagnose the cause and adjust or backtrack

Each step:
[REASON]
- Diff: compare current state (from STATE k-1) with goal and list all differences (Diff)
- Operator: pick one action to reduce a gap; judge whether it is directly solvable
- Subgoal: what blocks the operator? State the blocker.
  First, list each operand's per-unit contribution to the result.
  Then set up equations, solve for parameters, and confirm all parts
  are fully derived and the action moves closer to the goal.
  Repeat (Subgoal -> Operator) until a solvable operator is found.
[ACTION] State the solvable operator with its computed parameter
STATE k: Write out the FULL new state (k is the step number).
  You may show brief arithmetic before the STATE line when computing
  new compositions for mixtures.

### Solving guidelines (important for efficiency):
- When computing action parameters, set up equations and solve algebraically
  (e.g. Gaussian elimination or substitution). Do NOT guess-and-check with
  trial values.
- Solve for ALL goal conditions before starting STEP 1. Do not solve one
  goal, execute its steps, then solve the next goal.
- Before solving each target, globally consider which operands can
  contribute and whether one alone suffices or a combination is needed.
- Verify the solution ONCE by substituting back. If it checks out, proceed
  immediately. Do NOT compare alternative solutions or re-derive.
- Round numeric parameters to 3 decimal places.

End your solution with a "Plan summary:" section listing only the actions from your successful path.
"""

    DEFAULT_EXAMPLE = """\
Initial state: V1 (capacity 1L): 1 L L1; V2 (capacity 1L): 1 L L2; V3 (capacity 2L): empty
Goal state: V3: 1 L, 50% L1, 50% L2

STATE 0: V1 (capacity 1L): 1 L L1; V2 (capacity 1L): 1 L L2; V3 (capacity 2L): empty

STEP 1:
[REASON]
- Diff: V3 empty, goal 0.5L L1 + 0.5L L2 | V1 excess 0.5L | V2 excess 0.5L
- Operator: Pour to V3 — not solvable (V3 needs L1 + L2; source unknown)
- Subgoal: determine sources. Sources with L1: V1 only. Sources with L2: V2 only.
  No shared components across sources → solve one at a time. Start with L1.
  -> Operator: Pour from V1 to V3 — not solvable (amount unknown)
  -> Subgoal: V3 needs (1L * 50% = 0.5L) L1; V1 is pure L1 (concentration = 100%) → pour amount = 0.5 / 1.0 = 0.5L
    -> Operator: Pour 0.5L from V1 to V3 — solvable (V1 has 1L >= 0.5L, V3 cap 2L >= 0.5L)
[ACTION] Pour 0.5 L from V1 to V3
STATE 1:
  V1: 1 - 0.5 = 0.5L L1
  V3 receives 0.5L pure L1: L1 = 0.5*100% = 0.5L; total = 0.5L 
  V1: 0.5L L1 | V2: 1L L2 | V3: 0.5L L1

STEP 2:
[REASON]
- Diff: V3 has 0.5L L1 + 0L L2, goal 0.5L L1 + 0.5L L2 → remaining: 0.5L L2
- Operator: Pour to V3 — not solvable (amount unknown)
- Subgoal: Sources with L2: V2 only (pure L2). Amount unknown.
  V3 needs (1L * 50% = 0.5L) L2; V2 is pure L2 (concentration = 100%) → pour amount = 0.5 / 1.0 = 0.5L
  -> Operator: Pour 0.5L from V2 to V3 — solvable (V2 has 1L >= 0.5L, V3 cap 2L >= 0.5L)
[ACTION] Pour 0.5 L from V2 to V3
STATE 2:
  V2: 1 - 0.5 = 0.5L L2
  V3 receives 0.5L pure L2: L2 = 0.5*100% = 0.5L; total = 0.5 + 0.5 = 1L, L1 = 0.5/1 = 50%, L2 = 0.5/1 = 50% 
  V1: 0.5L L1 | V2: 0.5L L2 | V3: 1L 50%L1 50%L2

Plan summary:
1. Pour 0.5 L from V1 to V3
2. Pour 0.5 L from V2 to V3
"""

    def __init__(self, domain_description: Optional[str] = None):
        self.domain_description = domain_description or LiquidWorldDomain.get_domain_description()
        self.example = self.DEFAULT_EXAMPLE

    def set_example(self, example: str) -> None:
        """Replace the default few-shot example."""
        self.example = example

    def build_prompt(self, initial_state: str, goal_state: str) -> str:
        """Build the complete MEA prompt for the given problem."""
        return f"""# MEA Planning

## Domain Description:
{self.domain_description}

## Planning Task:
Given an initial state and a goal state, create a plan to transform the initial state into the goal state.
Use the MEA planning format below.

{self.STRUCTURED_FORMAT}
## Example:
{self.example}
---

## Problem to Solve:

Initial state: {initial_state}
Goal state: {goal_state}

STATE 0: {initial_state}

Solve this step by step using the MEA planning format above. Remember:
1. For each step, use the MEA loop inside [REASON], then [ACTION] -> STATE k:
2. End with a "Plan summary:" section

Start your solution:
"""


class Medaka:
    """Main implementation of Medaka (MEA-based LLM planning)."""

    def __init__(self, provider: str, *,
                 reasoning_effort: Optional[str] = None,
                 enable_thinking: Optional[bool] = None):
        from src.models.llm_client import get_llm_client
        self.provider = provider
        self.llm_client = get_llm_client(provider, reasoning_effort=reasoning_effort,
                                          enable_thinking=enable_thinking)
        self.prompt_builder = MedakaPromptBuilder()

    def get_prompt(self, initial_state: str, goal_state: str) -> str:
        """Return the prompt without calling the LLM."""
        from src.utils.config import Config
        prompt = self.prompt_builder.build_prompt(initial_state, goal_state)
        if Config.is_claude(self.provider):
            prompt += (
                "\nYour FIRST line of output MUST be 'STEP 1:'. "
                "Do NOT include any preamble, analysis, or equation solving before STEP 1."
            )
        return prompt

    def solve(self, initial_state: str, goal_state: str, *,
              temperature: float = 0.0,
              max_tokens: Optional[int] = None) -> Tuple[bool, List[str], str, str, dict]:
        """Solve a problem end-to-end: prompt -> LLM -> parse -> validate.

        Returns:
            (success, plan_steps, message, raw_output, usage)
        """
        import time
        from src.eval.plan_val import validate

        _PARSE_RETRIES = 2
        _EMPTY_RESP_RETRIES = 2
        _EMPTY_RESP_BACKOFF = 3.0

        from src.utils.config import Config
        prompt = self.get_prompt(initial_state, goal_state)
        max_tokens = max_tokens or Config.get_max_tokens()
        total_usage = {"input": 0, "output": 0, "total": 0, "reasoning": 0}
        empty_retries_left = _EMPTY_RESP_RETRIES
        parse_retries_left = _PARSE_RETRIES

        while True:
            output, usage = self.llm_client.generate(prompt, temperature, max_tokens=max_tokens)
            total_usage = {k: total_usage.get(k, 0) + usage.get(k, 0)
                           for k in ("input", "output", "total", "reasoning")}
            plan_steps = self.extract_plan(output)
            if plan_steps:
                break
            if not usage.get("total"):
                if empty_retries_left > 0:
                    empty_retries_left -= 1
                    print(f"  [empty-response retry {_EMPTY_RESP_RETRIES - empty_retries_left}/{_EMPTY_RESP_RETRIES}] "
                          f"0 tok, retrying in {_EMPTY_RESP_BACKOFF:.0f}s...")
                    time.sleep(_EMPTY_RESP_BACKOFF)
                    continue
                break
            if parse_retries_left > 0:
                parse_retries_left -= 1
                print(f"  [parse retry {_PARSE_RETRIES - parse_retries_left}/{_PARSE_RETRIES}] "
                      f"0 steps parsed, retrying...")
                continue
            break

        success, report = validate(initial_state, goal_state, plan_steps, verbose=False)
        message = report[-1] if report else ("PASS" if success else "FAIL")
        return success, plan_steps, message, output, total_usage

    def extract_plan(self, output: str) -> List[str]:
        """Extract plan steps from MEA output."""
        from src.utils.plan_parser import extract_pour_actions, find_last_plan_summary, normalize_plan_text

        if not isinstance(output, str):
            return []

        output = normalize_plan_text(output)
        steps = []

        plan_text = find_last_plan_summary(output)
        if plan_text:
            for line in plan_text.strip().split("\n"):
                line = re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip()
                normalized = re.sub(r"\band\s+([\d./]+\s*L\s+from)", r"Pour \1", line)
                pours = extract_pour_actions(normalized)
                if pours:
                    steps.extend(pours)
            if steps:
                return steps

        for m in re.finditer(r"\[(?:SELECT|ACTION)\]\s*(Pour .+)", output):
            raw = m.group(1).strip()
            pours = extract_pour_actions(raw)
            steps.extend(pours)
        return steps



def build_prompt(initial_state_str: str, goal_state_str: str) -> str:
    """Convenience wrapper kept for backward compatibility."""
    return MedakaPromptBuilder().build_prompt(initial_state_str, goal_state_str)
