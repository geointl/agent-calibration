"""Single orchestrator agent for NorSand calibration with iterative test suggestions."""

import os
from pathlib import Path

import yaml

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from src.calibration.agent.tools import (
    LLM_MODEL,
    TEMPLATE_CONFIG_PATH,
    parse_report,
    write_config,
    run_calibration,
    read_diagnostics,
    generate_virtual_test,
    screen_test,
    ask_human,
)

PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(variant: str) -> str:
    """Load system prompt from the prompts directory."""
    path = PROMPTS_DIR / f"{variant}.txt"
    return path.read_text()


def _template_has_ground_truth() -> bool:
    """True for synthetic studies: the template knows the data-generating parameters."""
    with open(TEMPLATE_CONFIG_PATH) as f:
        parameters = yaml.safe_load(f)["OptimizationSettings"]["parameters"]
    return all(spec.get("ground_truth") is not None
               for section in parameters.values() for spec in section.values() if spec["optimize"])


def create_calibration_agent():
    """Create the NorSand calibration orchestrator agent.

    With a template that holds ground-truth parameters (synthetic study), the
    agent generates the response of each proposed test itself. Otherwise the
    data are measurements and the agent asks the human for the new test's data.

    Setting NORSAND_SCREEN_RHO_MIN enables the sensitivity screen of each
    proposed test (screen_test tool and its prompt section).

    Returns:
        A LangGraph react agent with calibration tools and in-memory
        checkpointer (required for interrupt-based human-in-the-loop).
    """
    prompt = _load_prompt("full")
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    tools = [parse_report, write_config, run_calibration, read_diagnostics, ask_human]
    if _template_has_ground_truth():
        tools.insert(-1, generate_virtual_test)
    else:
        prompt += _load_prompt("measured_data")
    if os.getenv("NORSAND_SCREEN_RHO_MIN"):
        tools.insert(-1, screen_test)
        prompt += _load_prompt("screen")
    checkpointer = MemorySaver()
    return create_react_agent(llm, tools, prompt=prompt, checkpointer=checkpointer)
