"""Run the NorSand calibration agent on a triaxial test report.

Examples:
    # Paper case 1 (synthetic; the agent simulates the tests it proposes)
    python src/calibration/agent/run_agent.py

    # Your own test: a PDF report with tabulated stress-strain data
    python src/calibration/agent/run_agent.py --report my_test.pdf \\
        --template src/calibration/configs/config.yaml

    # Report (conditions) and curve data in separate files
    python src/calibration/agent/run_agent.py --report my_test.txt --data my_test.tsv \\
        --template src/calibration/configs/config.yaml
"""

import argparse
import os
import sys
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from langgraph.types import Command

EXAMPLE_DIR = PROJECT_ROOT / "data/synthetic"


def _extract_rounds_from_messages(messages):
    """Extract round-by-round calibration summaries from agent messages.

    A round is bounded by write_config -> run_calibration ->
    read_diagnostics.  The proposal reasoning (from the subsequent
    ask_human) is captured when it follows diagnostics.

    Args:
        messages: List of LangGraph message objects.

    Returns:
        List of round dicts with keys: round, new_test_conditions,
        calibration_result, diagnostics, proposal_reasoning.
    """
    rounds = []
    current_round = None

    for msg in messages:
        # AI tool-call messages define round boundaries
        if msg.type == "ai" and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc["name"]
                args = tc["args"]

                if name == "write_config":
                    specs = json.loads(args.get("test_specs_json", "[]"))
                    current_round = {
                        "round": args.get("round_number", len(rounds) + 1),
                        "new_test_conditions": specs,
                        "calibration_result": None,
                        "diagnostics": None,
                        "proposal_reasoning": None,
                    }
                    rounds.append(current_round)

                elif (name == "ask_human"
                      and current_round
                      and current_round.get("diagnostics")):
                    current_round["proposal_reasoning"] = args.get(
                        "message", "")

        # Capture AI-content reasoning not wrapped in ask_human
        # (e.g. final report when agent stops without proposing more tests)
        if (msg.type == "ai"
                and getattr(msg, "content", "")
                and not (hasattr(msg, "tool_calls") and msg.tool_calls)
                and current_round
                and current_round.get("diagnostics")
                and current_round["proposal_reasoning"] is None):
            current_round["proposal_reasoning"] = msg.content

        # Tool result messages fill in calibration / diagnostics results
        if msg.type == "tool" and current_round:
            content = getattr(msg, "content", "")
            if ("Calibration completed" in content
                    or "Calibration FAILED" in content):
                current_round["calibration_result"] = content
            elif "Fisher Matrix" in content:
                current_round["diagnostics"] = content

    return rounds


def _print_round_summary(round_info):
    """Print a compact summary for a completed calibration round.

    Proposal reasoning is omitted here because the interrupt handler
    already displays the full ask_human message to the operator.
    """
    print(f"\n{'─' * 60}")
    print(f"  ROUND {round_info['round']} SUMMARY")
    print(f"{'─' * 60}")

    if round_info["new_test_conditions"]:
        print("Test conditions:")
        for i, tc in enumerate(round_info["new_test_conditions"]):
            print(f"  [{i+1}] drainage={tc['drainage']}, "
                  f"sigM0={tc['sigM0']} kPa, e0={tc['e0']}")

    if round_info["calibration_result"]:
        print("\nCalibration:")
        for line in round_info["calibration_result"].split("\n"):
            print(f"  {line}")

    if round_info["diagnostics"]:
        print("\nDiagnostics:")
        for line in round_info["diagnostics"].split("\n"):
            print(f"  {line}")

    print(f"{'─' * 60}")


def _handle_interrupts(agent, config):
    """Poll for interrupt events and collect human input until the agent
    finishes or no more interrupts remain.

    After each resume, any newly completed calibration rounds are printed
    as summaries before the next interrupt message is displayed.
    """
    printed_rounds = 0

    while True:
        state = agent.get_state(config)
        messages = state.values.get("messages", [])

        # Print summaries for any newly completed rounds
        rounds = _extract_rounds_from_messages(messages)
        for r in rounds[printed_rounds:]:
            if r["calibration_result"]:
                _print_round_summary(r)
        printed_rounds = sum(1 for r in rounds if r["calibration_result"])

        if not state.next:
            return

        # Display all pending interrupt messages
        for task in state.tasks:
            if not hasattr(task, "interrupts"):
                continue
            for intr in task.interrupts:
                print("\n" + "=" * 60)
                print("[AGENT -> HUMAN]")
                msg = intr.value
                if isinstance(msg, dict):
                    msg = msg.get("message", str(msg))
                print(msg)
                print("=" * 60)

        human_input = input("Your response: ")
        agent.invoke(Command(resume=human_input), config=config)


def save_agent_trace(agent, config, output_dir):
    """Extract the full agent decision trace from LangGraph state and save as JSON.

    Serializes every message (user input, agent reasoning, tool calls,
    tool results) into a structured list for post-hoc analysis and
    paper figures.

    Args:
        agent: The LangGraph agent instance.
        config: LangGraph config dict (with thread_id).
        output_dir: Directory to write the trace JSON file.

    Returns:
        Path to the saved trace file.
    """
    state = agent.get_state(config)
    messages = state.values.get("messages", [])

    trace = []
    for msg in messages:
        entry = {"type": msg.type, "content": getattr(msg, "content", "")}

        # Agent messages carry tool call decisions
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            entry["tool_calls"] = [
                {"name": tc["name"], "args": tc["args"]}
                for tc in msg.tool_calls
            ]

        # Tool results link back to which call they answer
        if hasattr(msg, "tool_call_id"):
            entry["tool_call_id"] = msg.tool_call_id

        trace.append(entry)

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_path = os.path.join(output_dir, f"agent_trace_{timestamp}.json")
    with open(trace_path, "w") as f:
        json.dump(trace, f, indent=2, default=str)

    return trace_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", default=str(PROJECT_ROOT / "data/example_report_case01.txt"),
                        help="triaxial test report (.pdf, .txt or .md)")
    parser.add_argument("--data", default=None,
                        help="curve data (.tsv or .json); omit when the report tabulates the curves")
    parser.add_argument("--template", default=str(EXAMPLE_DIR / "case01_undrained_loose_p100_calibration.yaml"),
                        help="calibration template: initial guesses, bounds and solver settings")
    parser.add_argument("--screen-rho-min", type=float, default=None,
                        help="enable the sensitivity screen of proposed tests with this rho_min, e.g. 3.0")
    args = parser.parse_args()
    if args.data is None and args.report == parser.get_default("report"):
        args.data = str(EXAMPLE_DIR / "case01_undrained_loose_p100.tsv")

    # The tools read these when imported.
    os.environ["NORSAND_TEMPLATE_CONFIG"] = str(Path(args.template).resolve())
    if args.screen_rho_min is not None:
        os.environ["NORSAND_SCREEN_RHO_MIN"] = str(args.screen_rho_min)

    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not found. Set it in the environment or in a .env file "
                 "at the repository root (see .env.example).")

    from src.calibration.agent.agent import create_calibration_agent
    from src.calibration.agent.tools import AGENT_LOG_DIR

    agent = create_calibration_agent()
    config = {
        "configurable": {"thread_id": "calibration-session"},
        "recursion_limit": 150,
    }

    data_line = (f"**Data file:** {Path(args.data).resolve()}\n\n" if args.data else
                 "**Data file:** none given; use the data_path returned by parse_report.\n\n")
    user_message = (
        f"Calibrate NorSand parameters from this triaxial test report:\n\n"
        f"**Report:** {Path(args.report).resolve()}\n\n"
        f"{data_line}"
        f"**Solver:** lbfgsb, 50 iterations\n\n"
        f"Run the full calibration loop: parse, calibrate, diagnose, "
        f"and iteratively suggest additional tests until parameters are "
        f"well-identified (or max 3 total tests)."
    )

    print("\n" + "=" * 60)
    print("Starting Calibration Agent")
    print("=" * 60 + "\n")
    agent.invoke({"messages": [("user", user_message)]}, config=config)
    _handle_interrupts(agent, config)

    # Print the agent's final message
    state = agent.get_state(config)
    messages = state.values.get("messages", [])
    for msg in reversed(messages):
        if msg.type == "ai" and getattr(msg, "content", ""):
            print("\n" + "=" * 60)
            print("FINAL CALIBRATION REPORT")
            print("=" * 60)
            print(msg.content)
            break

    # Save full trace and round-structured decision summary
    trace_path = save_agent_trace(agent, config, str(AGENT_LOG_DIR))
    print(f"\nAgent trace saved to {trace_path}")

    summary = _extract_rounds_from_messages(messages)
    summary_path = trace_path.replace("agent_trace_", "decision_summary_")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Decision summary saved to {summary_path}")


if __name__ == "__main__":
    main()
