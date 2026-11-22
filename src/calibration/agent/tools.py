"""Tool definitions for the NorSand calibration agent."""

import os
import re
import json
import pickle
import yaml
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt

from src.calibration.utils import load_test_data

# Project root for resolving config templates and output paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Template YAML: initial guesses, bounds, solver settings (and ground truth for synthetic studies).
TEMPLATE_CONFIG_PATH = Path(
    os.getenv("NORSAND_TEMPLATE_CONFIG",
              str(PROJECT_ROOT / "src/calibration/configs/config.yaml"))
)
AGENT_LOG_DIR = Path(os.getenv("NORSAND_AGENT_LOG_DIR", str(PROJECT_ROOT / "logs/agent_run")))
LLM_MODEL = os.getenv("NORSAND_LLM_MODEL", "gpt-4o")


# ---------------------------------------------------------------------------
# Pydantic schema for structured report parsing
# ---------------------------------------------------------------------------

class CurveColumns(BaseModel):
    """Location of the measured curves among the numeric tables of a report."""
    table: int = Field(description="Index of the numeric table that holds the measured curves")
    eps1: int = Field(description="0-based column index of axial strain")
    q: Optional[int] = Field(default=None, description="Column of deviatoric stress q")
    ue: Optional[int] = Field(default=None, description="Column of excess pore pressure")
    epsV: Optional[int] = Field(default=None, description="Column of volumetric strain")
    p: Optional[int] = Field(default=None, description="Column of mean effective stress p'")
    strain_unit: Literal["percent", "fraction"] = Field(description="Unit of the strain columns")
    stress_unit: Literal["kPa", "MPa", "psi"] = Field(description="Unit of the stress columns")
    epsV_contraction_positive: bool = Field(
        default=True, description="True if volumetric strain is positive in contraction")


class ParsedTestParams(BaseModel):
    """Triaxial test parameters extracted from a geotechnical report."""
    drainage: str = Field(description="Drainage condition: 'drained' or 'undrained'")
    sigM0: Optional[float] = Field(default=None, description="Initial mean effective stress (kPa)")
    e0: Optional[float] = Field(default=None, description="Initial void ratio")
    rationale: Optional[str] = Field(
        default=None,
        description=(
            "For each filled field, whether it was stated in the report or "
            "inferred, and the quantities and reasoning used."
        ),
    )
    curve: Optional[CurveColumns] = Field(
        default=None,
        description="Columns of the measured curves; null if no listed numeric table holds them.",
    )


def _resolve_report_text(report: str) -> str:
    """Return report contents. A one-line existing file path is read (PDF via OCR if needed)."""
    raw = report.strip()
    if "\n" in raw:
        return report
    path = Path(raw)
    if not path.is_file():
        return report
    if path.suffix.lower() == ".pdf":
        return _pdf_to_text(path)
    return path.read_text(encoding="utf-8", errors="replace")


def _pdf_to_text(path: Path) -> str:
    """Digital text layer first; OCR a scanned PDF if the layer is empty."""
    text = _pdf_text_layer(path)
    if len(text.strip()) >= 400:
        return text
    ocr = _pdf_ocr(path)
    if ocr.strip():
        return ocr
    if text.strip():
        return text
    raise RuntimeError(
        f"No text extracted from {path}. Install pypdf (digital PDFs) or "
        "pdf2image and pytesseract (scanned PDFs)."
    )


def _pdf_text_layer(path: Path) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except ImportError:
        pass
    try:
        import subprocess
        out = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            capture_output=True, text=True, check=False,
        )
        if out.returncode == 0:
            return out.stdout
    except FileNotFoundError:
        pass
    return ""


def _pdf_ocr(path: Path) -> str:
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError:
        return ""
    pages = convert_from_path(str(path), dpi=200)
    return "\n".join(pytesseract.image_to_string(image) for image in pages)


_NUMBER = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")
_TO_KPA = {"kPa": 1.0, "MPa": 1000.0, "psi": 6.894757}


def _numeric_tables(text: str, min_rows: int = 5) -> list[dict]:
    """Blocks of consecutive lines holding the same number (>= 2) of numeric tokens.

    Numbers are read from the text as written, never re-typed by the LLM.
    Returns [{"header": last text line before the block, "rows": [[float, ...], ...]}].
    """
    # shortcut: commas, semicolons and pipes are column separators, so "1,234.5"
    # reads as two numbers; handle thousands separators if such reports show up.
    tables, rows, header, last_text = [], [], "", ""
    for line in text.splitlines() + [""]:
        tokens = re.sub(r"[,;|]", " ", line).split()
        numeric = len(tokens) >= 2 and all(_NUMBER.match(t) for t in tokens)
        if rows and not (numeric and len(tokens) == len(rows[0])):
            if len(rows) >= min_rows:
                tables.append({"header": header, "rows": rows})
            rows = []
        if numeric:
            if not rows:
                header = last_text
            rows.append([float(t) for t in tokens])
        elif line.strip():
            last_text = line.strip()
    return tables


def _write_test_json(parsed: dict, tables: list[dict], name: str) -> tuple[Optional[str], str]:
    """Write the extracted test (conditions + curves, solver units) as JSON.

    Returns (path or None, note for the operator).
    """
    curve = parsed.get("curve")
    if not curve or not 0 <= curve["table"] < len(tables):
        return None, ("No tabulated curve data found in the report (curves shown only as "
                      "figures are not digitized). Ask the human for a data file.")
    first = tables[curve["table"]]
    width = len(first["rows"][0])
    columns = {k: curve[k] for k in ("eps1", "q", "ue", "epsV", "p") if curve.get(k) is not None}
    if any(not 0 <= c < width for c in columns.values()):
        return None, f"Column mapping {columns} does not fit the {width}-column table. Ask the human."
    # A table continued on later pages comes as further blocks of the same width
    # whose axial strain keeps increasing.
    rows = list(first["rows"])
    for t in tables[curve["table"] + 1:]:
        if len(t["rows"][0]) != width or t["rows"][0][curve["eps1"]] < rows[-1][curve["eps1"]]:
            break
        rows += t["rows"]
    strain = 100.0 if curve["strain_unit"] == "fraction" else 1.0
    factor = {"eps1": strain, "epsV": strain * (1 if curve["epsV_contraction_positive"] else -1)}
    data = {k: [r[c] * factor.get(k, _TO_KPA[curve["stress_unit"]]) for r in rows]
            for k, c in columns.items()}
    if len(columns) < 2:
        return None, "Only axial strain was identified in the table. Ask the human for a data file."
    out = AGENT_LOG_DIR / f"{name}_data.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"drainage": parsed["drainage"], "sigM0": parsed["sigM0"], "e0": parsed["e0"],
                   "units": {"eps1": "%", "epsV": "%", "q": "kPa", "ue": "kPa", "p": "kPa"},
                   "data": data}, f, indent=1)
    ranges = ", ".join(f"{k}: {min(v):.4g} to {max(v):.4g}" for k, v in data.items())
    return str(out), f"{len(rows)} points extracted from the report table ({ranges})."


# ---------------------------------------------------------------------------
# Tool 1: parse_report
# ---------------------------------------------------------------------------

@tool
def parse_report(report_text: str) -> dict:
    """Parse a free-form geotechnical report: test conditions and measured curves.

    Args:
        report_text: Report text, or a path to a .pdf/.txt/.md file. PDFs with
            no text layer are OCR'd when pdf2image and pytesseract are available.

    Returns:
        Dict with keys: drainage, sigM0, e0, rationale, data_path, data_note.
        data_path is a JSON data file extracted from the report's tabulated
        stress-strain data, ready to pass to write_config; it is null when the
        report has no such table (then ask the human for a data file).
    """
    source = Path(report_text.strip()) if "\n" not in report_text.strip() else None
    name = source.stem if source is not None and source.is_file() else "report"
    report_text = _resolve_report_text(report_text)
    tables = _numeric_tables(report_text)
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    structured_llm = llm.with_structured_output(ParsedTestParams)

    system_prompt = (
        "You are a geotechnical engineering expert. Extract triaxial test "
        "parameters from the report below.\n"
        "Target fields:\n"
        "- Drainage condition ('drained' or 'undrained')\n"
        "- Initial mean effective stress sigM0 in kPa "
        "(may appear under another name, e.g. consolidation or cell pressure)\n"
        "- Initial void ratio e0\n\n"
        "Prefer values stated in the report.\n"
        "If a target value is not stated, infer it from other quantities in "
        "the report using standard geotechnical knowledge. Do not invent a "
        "value from a qualitative label alone. If you cannot infer a value, "
        "leave it null.\n"
        "In rationale, for every filled field, say whether it was stated or "
        "inferred and name the source quantities and reasoning.\n"
        "If several tests are described, extract the one the user highlighted, "
        "or the first clearly specified triaxial test.\n"
        "Default to 'undrained' if drainage is still unclear after inference."
    )

    if tables:
        system_prompt += (
            "\n\nThe numeric tables found in the report are listed after it. If one "
            "holds the measured curves of the extracted test (axial strain with "
            "deviatoric stress, excess pore pressure or volumetric strain), fill "
            "'curve' with its index, the 0-based column of each quantity, and the "
            "units. Map only columns that directly list these quantities; leave "
            "the others null."
        )
        report_text += "\n\nNumeric tables:\n" + "\n".join(
            f"[{i}] header: {t['header']!r}; {len(t['rows'])} rows; "
            f"first rows: {t['rows'][:3]}; last row: {t['rows'][-1]}"
            for i, t in enumerate(tables)
        )

    response = structured_llm.invoke(f"{system_prompt}\n\nReport:\n{report_text}")
    parsed = response.model_dump()
    parsed["data_path"], parsed["data_note"] = _write_test_json(parsed, tables, name)
    del parsed["curve"]
    return parsed


# ---------------------------------------------------------------------------
# Tool 2: write_config
# ---------------------------------------------------------------------------

@tool
def write_config(test_specs_json: str, data_paths_json: str,
                 solver: str = "lbfgsb", n_iterations: int = 50,
                 calibrated_params_json: str = "",
                 round_number: int = 1,
                 previous_config_path: str = "") -> str:
    """Write a calibration config YAML from test specifications.

    When previous_config_path is provided, the new test specs and data paths
    are APPENDED to the tests from the previous config. This ensures
    cumulative multi-test calibration without needing to re-specify old tests.

    Args:
        test_specs_json: JSON string of list of NEW test dicts, each with keys
            'drainage', 'sigM0', 'e0'. Example:
            '[{"drainage": "undrained", "sigM0": 100.0, "e0": 0.81}]'
        data_paths_json: JSON string of list of NEW data file paths (.tsv or
            the .json written by parse_report). Example: '["data/test1.tsv"]'
        solver: Optimization solver type ('lbfgsb' or 'adam').
        n_iterations: Number of optimization iterations.
        calibrated_params_json: Optional JSON string of previously calibrated
            parameters to use as warm-start initial guess, e.g.
            '{"Gamma": 0.92, "lambda": 0.04, "Mtc": 1.1, ...}'.
            Leave empty on the first round.
        round_number: Calibration round (1, 2, 3, ...). Each round gets its
            own log subdirectory to preserve history.
        previous_config_path: Optional path to the previous round's config
            YAML. When provided, existing tests and data paths are read from
            it and the new specs are appended automatically.

    Returns:
        Path to the generated config YAML file.
    """
    new_specs = json.loads(test_specs_json)
    new_paths = json.loads(data_paths_json)
    calibrated_params = json.loads(calibrated_params_json) if calibrated_params_json else {}

    # If previous config exists, read old tests and prepend them
    test_specs = []
    data_paths = []
    if previous_config_path and os.path.exists(previous_config_path):
        with open(previous_config_path, "r") as f:
            prev_cfg = yaml.safe_load(f)
        prev_ns = prev_cfg["NorSandConfig"]
        prev_opt = prev_cfg["OptimizationSettings"]

        # Reconstruct previous test specs from the saved config
        prev_drainage = prev_ns["simulation"]["drainage"]
        prev_soil = prev_ns["material_parameters"]["soil_state"]
        prev_e0 = prev_soil["e0"]
        prev_sigM0 = prev_soil["sigM0"]
        prev_data = prev_opt["data_path"]

        # Normalize to lists for uniform handling
        if not isinstance(prev_drainage, list):
            prev_drainage = [prev_drainage]
        if not isinstance(prev_e0, list):
            prev_e0 = [prev_e0]
        if not isinstance(prev_sigM0, list):
            prev_sigM0 = [prev_sigM0]
        if not isinstance(prev_data, list):
            prev_data = [prev_data]

        for d, e, s in zip(prev_drainage, prev_e0, prev_sigM0):
            test_specs.append({"drainage": d, "e0": e, "sigM0": s})
        data_paths.extend(prev_data)

    test_specs.extend(new_specs)
    data_paths.extend(new_paths)
    n_tests = len(test_specs)

    # Load template config
    with open(TEMPLATE_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    norsand = config["NorSandConfig"]
    opt = config["OptimizationSettings"]

    # --- Simulation settings ---
    norsand["simulation"]["batch_size"] = n_tests
    if n_tests == 1:
        norsand["simulation"]["drainage"] = test_specs[0]["drainage"]
    else:
        norsand["simulation"]["drainage"] = [t["drainage"] for t in test_specs]

    # --- Simulation window: the simulated axial strain ends somewhat short of
    # MaxStrainForSim (by an amount that depends on the parameters), so keep a
    # margin above the longest measured curve.
    if (opt.get("loss_interpolation") or {}).get("enabled"):
        obs_max = float(max(load_test_data(path)["eps1"].max() for path in data_paths)) / 100.0
        norsand["simulation"]["MaxStrainForSim"] = max(
            norsand["simulation"]["MaxStrainForSim"], round(1.25 * obs_max, 4))

    # --- Soil state from test specs ---
    soil = norsand["material_parameters"]["soil_state"]
    if n_tests == 1:
        soil["e0"] = test_specs[0]["e0"]
        soil["sigM0"] = test_specs[0]["sigM0"]
    else:
        soil["e0"] = [t["e0"] for t in test_specs]
        soil["sigM0"] = [t["sigM0"] for t in test_specs]
    soil["eta0"] = 0.0

    # --- Inject ground truth for non-optimized material parameters ---
    for section_name, section_params in opt["parameters"].items():
        norsand_section = norsand["material_parameters"].get(section_name, {})
        for param_name, settings in section_params.items():
            if not settings["optimize"]:
                gt = settings.get("ground_truth")
                if gt is not None and param_name not in ("e0", "sigM0", "eta0"):
                    norsand_section[param_name] = gt

    # --- Warm-start: inject previously calibrated params as initial guess ---
    if calibrated_params:
        for section in norsand["material_parameters"].values():
            for param_name in section:
                if param_name in calibrated_params:
                    section[param_name] = calibrated_params[param_name]

    # --- Set loss_keys based on drainage conditions ---
    drainages = set(t["drainage"].lower() for t in test_specs)
    has_undrained = "undrained" in drainages
    has_drained = "drained" in drainages
    if has_undrained and has_drained:
        opt["loss_keys"] = ["q", "ue", "epsV"]
    elif has_drained:
        opt["loss_keys"] = ["q", "epsV"]
    else:
        opt["loss_keys"] = ["q", "ue"]

    # --- Optimization settings ---
    opt["solver"] = solver
    opt["n_iterations"] = n_iterations
    opt["identifiability_diagnostics"] = {"enabled": True}
    log_dir = str(AGENT_LOG_DIR / f"round_{round_number}")
    opt["log_dir"] = log_dir

    # Data paths (single string or list)
    opt["data_path"] = data_paths if n_tests > 1 else data_paths[0]

    # Update ground truth for soil_state to match current test specs
    soil_gt = opt["parameters"]["soil_state"]
    soil_gt["e0"]["ground_truth"] = soil["e0"]
    soil_gt["sigM0"]["ground_truth"] = soil["sigM0"]

    # Write config
    os.makedirs(log_dir, exist_ok=True)
    output_path = os.path.join(log_dir, "config.yaml")
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return output_path


# ---------------------------------------------------------------------------
# Tool 3: run_calibration
# ---------------------------------------------------------------------------

@tool
def run_calibration(config_path: str) -> str:
    """Run the calibration solver and return a summary of results.

    Args:
        config_path: Path to the calibration config YAML.

    Returns:
        Summary string with calibrated parameters and final loss.
    """
    from src.calibration.run_solver import main as run_solver_main

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    log_dir = cfg["OptimizationSettings"]["log_dir"]

    try:
        run_solver_main(cfg)
    except Exception as e:
        return f"Calibration FAILED: {e}"

    # Read results from saved pickle
    results_path = os.path.join(log_dir, "results.pkl")
    with open(results_path, "rb") as f:
        history = pickle.load(f)

    # Extract final iteration results
    final_iter = max(history.keys())
    final = history[final_iter]
    data_loss = final["data_loss"]
    params = {k: v.item() if hasattr(v, "item") else v
              for k, v in final["material_params"].items()}

    lines = [
        f"Calibration completed ({final_iter + 1} iterations).",
        f"Final data loss: {data_loss:.6e}",
        f"Log directory: {log_dir}",
        "Calibrated parameters:",
    ]
    for name, val in params.items():
        lines.append(f"  {name}: {val:.6f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 4: read_diagnostics
# ---------------------------------------------------------------------------

@tool
def read_diagnostics(log_dir: str) -> str:
    """Read Fisher/Jacobian identifiability diagnostics from a completed calibration.

    Expects identifiability_diagnostics.pkl in the log directory (written when
    OptimizationSettings.identifiability_diagnostics.enabled is true).

    Args:
        log_dir: Path to the calibration log directory.

    Returns:
        Formatted summary of Fisher matrix diagnostics including eigenvalue
        spectrum and weakest parameter directions.
    """
    pkl_path = os.path.join(log_dir, "identifiability_diagnostics.pkl")
    if not os.path.exists(pkl_path):
        return (
            f"No diagnostics found at {pkl_path}. "
            "Is identifiability_diagnostics.enabled set in the config?"
        )

    with open(pkl_path, "rb") as f:
        diag = pickle.load(f)

    evals = diag["eigenvalues"].tolist()
    param_names = diag["param_names"]
    weak_ranked = diag["weak_params_ranked"]

    emin, emax = evals[0], evals[-1]

    lines = [
        "=== Fisher Matrix Identifiability Diagnostics ===",
        f"Parameters: {param_names}",
        f"Min eigenvalue: {emin:.4e}",
        f"Max eigenvalue: {emax:.4e}",
        "",
        "Eigenvalue spectrum (ascending):",
    ]
    for i, ev in enumerate(evals):
        lines.append(f"  [{i}] {ev:.4e}")

    lines.append("")
    lines.append("Weakest direction contributions (highest = most poorly identified):")
    for name, contrib in weak_ranked:
        lines.append(f"  {name}: {contrib:.2%}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 5: generate_virtual_test
# ---------------------------------------------------------------------------

@tool
def generate_virtual_test(e0: float, sigM0: float, drainage: str,
                          output_name: str) -> str:
    """Generate synthetic triaxial test data using ground truth parameters.

    Simulates what a real lab test would produce at the given conditions.

    Args:
        e0: Initial void ratio for the virtual test.
        sigM0: Confining pressure in kPa.
        drainage: 'drained' or 'undrained'.
        output_name: Name for the output file (without extension).

    Returns:
        Path to the generated TSV data file.
    """
    from src.calibration.generate_synthetic_data import generate_single_test

    # Load ground truth material parameters from the template config
    with open(TEMPLATE_CONFIG_PATH, "r") as f:
        template = yaml.safe_load(f)

    opt_params = template["OptimizationSettings"]["parameters"]
    ground_truth = {}
    for section_name, section_params in opt_params.items():
        ground_truth[section_name] = {}
        for param_name, settings in section_params.items():
            gt = settings.get("ground_truth")
            # Use scalar ground truth only (skip list values like e0, sigM0)
            if gt is not None and not isinstance(gt, list):
                ground_truth[section_name][param_name] = gt

    sim = template["NorSandConfig"]["simulation"]
    output_path = str(AGENT_LOG_DIR / f"{output_name}.tsv")
    generate_single_test(ground_truth, e0=e0, sigM0=sigM0, drainage=drainage,
                         output_path=output_path, max_strain=sim["MaxStrainForSim"],
                         num_pts=sim["NumTheoPts"])
    return output_path


# ---------------------------------------------------------------------------
# Optional tool: screen_test (enabled when NORSAND_SCREEN_RHO_MIN is set)
# ---------------------------------------------------------------------------

SCREEN_MIN_SHARE = 0.10  # parameters with at least this share of the weakest direction are swept


@tool
def screen_test(log_dir: str, e0: float, sigM0: float, drainage: str) -> str:
    """Screen a proposed follow-up test before asking the human to approve it.

    Simulates the proposed test while sweeping each parameter that dominates the
    weakest Fisher direction over its bounds, and compares the response spread
    with a nominal measurement noise (rho). STOP means the test cannot
    distinguish these parameters from noise.

    Args:
        log_dir: Log directory of the latest calibration round.
        e0: Proposed initial void ratio.
        sigM0: Proposed confining pressure in kPa.
        drainage: Proposed drainage, 'drained' or 'undrained'.

    Returns:
        rho per swept parameter and response channel, and the decision.
    """
    from src.calibration.identifiability_diagnostics import screen_rho

    rho_min = float(os.environ["NORSAND_SCREEN_RHO_MIN"])
    with open(os.path.join(log_dir, "calibrated_params.json"), "r") as f:
        saved = json.load(f)
    with open(os.path.join(log_dir, "identifiability_diagnostics.pkl"), "rb") as f:
        weak = pickle.load(f)["weak_params_ranked"]
    bounds = {p: s["bounds"] for sec in saved["OptimizationSettings"]["parameters"].values()
              for p, s in sec.items() if s.get("optimize")}
    channels = ["q", "ue"] if drainage == "undrained" else ["q", "epsV"]
    test = {"e0": e0, "sigM0": sigM0, "drainage": drainage}

    lines = [f"Sensitivity screen (rho_min = {rho_min:g}):"]
    swept, blind = [], []
    for name, share in weak:
        if share < SCREEN_MIN_SHARE:
            break
        rho = screen_rho(saved["NorSandConfig"], test, name, bounds[name], channels)
        lines.append(f"  {name} ({share:.0%} of weakest direction): "
                     + ", ".join(f"rho_{k} = {v:.2f}" for k, v in rho.items()))
        swept.append(name)
        if max(rho.values()) < rho_min:
            blind.append(name)
    if blind == swept:
        lines.append(f"Decision: STOP. The proposed test cannot distinguish {', '.join(blind)} "
                     "from measurement noise; report them as weakly observable.")
    else:
        lines.append("Decision: PROCEED.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 6: ask_human
# ---------------------------------------------------------------------------

@tool
def ask_human(message: str) -> str:
    """Display a message to the human operator and wait for their response.

    Use this for config confirmation, approval of suggested follow-up tests, generating
    virtual test data, or corrections.

    Args:
        message: The message or question to show the human.

    Returns:
        The human's typed response.
    """
    # interrupt() pauses graph execution and returns control to the caller.
    # The caller collects human input and resumes with Command(resume=...).
    response = interrupt({"message": message})
    return response
