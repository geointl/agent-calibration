"""Calibration pipeline tests: data loading, loss grids, solver, report parser.

All tests run without an API key (the LLM is replaced by a stub) and use a
short, coarse simulation so the whole file takes seconds.
"""

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.calibration import solvers
from src.calibration.agent import tools
from src.calibration.generate_synthetic_data import generate_single_test
from src.calibration.identifiability_diagnostics import compute_fisher_from_solver
from src.calibration.solver_base import SolverBase
from src.calibration.utils import load_test_data

ROOT = Path(__file__).resolve().parents[1]
TRUTH = {
    "csl": {"Gamma": 0.95, "lambda": 0.05},
    "plasticity": {"Mtc": 1.2, "N": 0.3, "chi_tc": 2.5, "H0": 250.0, "Hy": 10.0},
    "elasticity": {"G_at_p0": 10000.0, "Gexp": 0.0, "nu": 0.2},
}
TESTS = [{"drainage": "undrained", "e0": 0.76, "sigM0": 100.0},
         {"drainage": "drained", "e0": 0.70, "sigM0": 300.0}]
MAX_STRAIN, NUM_PTS = 0.10, 300


@pytest.fixture
def cfg(tmp_path):
    """Two-test config starting at the true parameters, data on the model's step grid."""
    with open(ROOT / "src/calibration/configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    paths = [generate_single_test(TRUTH, output_path=str(tmp_path / f"test{i}.tsv"),
                                  max_strain=MAX_STRAIN, num_pts=NUM_PTS, **t)
             for i, t in enumerate(TESTS)]
    ns, opt = cfg["NorSandConfig"], cfg["OptimizationSettings"]
    ns["simulation"].update(MaxStrainForSim=MAX_STRAIN, NumTheoPts=NUM_PTS, batch_size=2,
                            drainage=[t["drainage"] for t in TESTS])
    for section, values in TRUTH.items():
        ns["material_parameters"][section].update(values)
    ns["material_parameters"]["soil_state"].update(e0=[t["e0"] for t in TESTS],
                                                    sigM0=[t["sigM0"] for t in TESTS])
    opt.update(data_path=paths, loss_keys=["q", "ue", "epsV"], log_dir=str(tmp_path / "log"),
               save_interval=10_000, n_iterations=3)
    return cfg


def _loss(cfg):
    solver = solvers.LBFGSBSolver(cfg, torch.device("cpu"))
    return solver, solver.data_loss_function(solver.forward(), "mse").item()


def _resample(cfg, tmp_path):
    """Rewrite the two tests on unrelated grids: a short coarse TSV and an irregular JSON with gaps."""
    a, b = (pd.read_csv(p, sep="\t") for p in cfg["OptimizationSettings"]["data_path"])
    a = a.iloc[: int(0.7 * len(a)) : 9][["eps1", "q", "ue"]]          # 24 points, stops early
    b = b.iloc[np.sort(np.random.default_rng(0).choice(len(b), 41, replace=False))][["eps1", "q", "epsV"]]
    b.iloc[5, b.columns.get_loc("epsV")] = np.nan                      # one unrecorded reading
    a.to_csv(tmp_path / "a.tsv", sep="\t", index=False)
    with open(tmp_path / "b.json", "w") as f:
        json.dump({"data": {k: [None if np.isnan(x) else x for x in v] for k, v in b.to_dict("list").items()}}, f)
    cfg["OptimizationSettings"]["data_path"] = [str(tmp_path / "a.tsv"), str(tmp_path / "b.json")]
    return len(a) * 2 + len(b) * 2 - 1  # observed values entering the loss


def test_json_and_tsv_load_the_same(tmp_path, cfg):
    path = cfg["OptimizationSettings"]["data_path"][0]
    df = pd.read_csv(path, sep="\t")
    with open(tmp_path / "t.json", "w") as f:
        json.dump({"drainage": "undrained", "data": df.to_dict("list")}, f)
    pd.testing.assert_frame_equal(load_test_data(tmp_path / "t.json"), load_test_data(path))


def test_interpolation_is_exact_at_nodes_and_linear_between():
    x = torch.tensor([0.0, 1.0, 1.0, 3.0], dtype=torch.float64)  # repeated strain must not give nan
    y = torch.tensor([0.0, 2.0, 2.0, 6.0], dtype=torch.float64)
    out = SolverBase._torch_interp(torch.tensor([0.0, 0.5, 1.0, 2.0, 3.0, 9.0], dtype=torch.float64), x, y)
    assert out.tolist() == [0.0, 1.0, 2.0, 4.0, 6.0, 6.0]


def test_interpolated_loss_accepts_tests_on_different_grids(tmp_path, cfg):
    n_obs = _resample(cfg, tmp_path)
    solver, loss_at_truth = _loss(cfg)
    assert loss_at_truth < 1e-20  # observed points are nodes of the true curves

    fisher = compute_fisher_from_solver(solver)
    assert fisher["J"].shape == (n_obs, 8) and torch.isfinite(fisher["eigenvalues"]).all()

    cfg["NorSandConfig"]["material_parameters"]["plasticity"]["Mtc"] = 1.3
    assert _loss(cfg)[1] > 1e-4


def test_step_aligned_loss_matches_interpolated_loss_at_truth_and_rejects_other_grids(tmp_path, cfg):
    cfg["OptimizationSettings"]["loss_interpolation"]["enabled"] = False
    assert _loss(cfg)[1] < 1e-20
    _resample(cfg, tmp_path)
    with pytest.raises(ValueError, match="loss_interpolation"):
        _loss(cfg)


def test_data_beyond_simulation_window_is_rejected(cfg):
    cfg["NorSandConfig"]["simulation"]["MaxStrainForSim"] = 0.05
    with pytest.raises(ValueError, match="simulation window"):
        _loss(cfg)


def test_number_of_data_files_must_match_batch_size(cfg):
    cfg["OptimizationSettings"]["data_path"] = cfg["OptimizationSettings"]["data_path"][:1]
    with pytest.raises(ValueError, match="batch_size"):
        _loss(cfg)


def test_lbfgsb_reduces_loss_and_is_deterministic(cfg):
    mp = cfg["NorSandConfig"]["material_parameters"]
    mp["plasticity"].update(Mtc=1.3, H0=300.0)
    mp["csl"]["Gamma"] = 0.98

    def run():
        solver = solvers.LBFGSBSolver(copy.deepcopy(cfg), torch.device("cpu"))
        history = solver.optimize()
        return history, {k: float(v) for k, v in solver.denormalize_parameters().items()}

    (h1, p1), (_, p2) = run(), run()
    assert p1 == p2  # same inputs, bit-identical result
    assert min(h["data_loss"] for h in h1.values()) < 0.5 * h1[0]["data_loss"]
    assert (Path(cfg["OptimizationSettings"]["log_dir"]) / "identifiability_diagnostics.pkl").exists()


# ---------------------------------------------------------------------------
# Report parser
# ---------------------------------------------------------------------------

REPORT = """Triaxial Test Report TX-9
Consolidated drained test, cell pressure 0.2 MPa, initial void ratio 0.68.
Sample height 100 mm, diameter 50 mm.

Axial strain (-)   q (MPa)   Volumetric strain (-), dilation positive
0.000   0.000    0.0000
0.005   0.110   -0.0010
0.010   0.190   -0.0015
0.020   0.280   -0.0012
0.040   0.350    0.0005
Page 2 of 2
0.080   0.380    0.0040
0.120   0.385    0.0060
0.160   0.385    0.0066
0.200   0.384    0.0068
0.220   0.384    0.0069
"""


@pytest.fixture
def stub_llm(monkeypatch, tmp_path):
    """Replace the LLM by the column mapping a correct reading of REPORT gives."""
    seen = {}

    class StubLLM:
        def __init__(self, **_):
            pass

        def with_structured_output(self, schema):
            return self

        def invoke(self, prompt):
            seen["prompt"] = prompt
            return tools.ParsedTestParams(
                drainage="drained", sigM0=200.0, e0=0.68, rationale="stated",
                curve=tools.CurveColumns(table=0, eps1=0, q=1, epsV=2, strain_unit="fraction",
                                         stress_unit="MPa", epsV_contraction_positive=False))

    monkeypatch.setattr(tools, "ChatOpenAI", StubLLM)
    monkeypatch.setattr(tools, "AGENT_LOG_DIR", tmp_path / "agent")
    return seen


def test_numeric_tables_are_read_from_text():
    tables = tools._numeric_tables(REPORT)
    assert [len(t["rows"]) for t in tables] == [5, 5]
    assert tables[0]["header"].startswith("Axial strain") and tables[1]["rows"][-1] == [0.22, 0.384, 0.0069]
    assert tools._numeric_tables("Height 100 50\nno table here 1 2 3") == []


def _check_extracted(result):
    assert (result["drainage"], result["sigM0"], result["e0"]) == ("drained", 200.0, 0.68)
    df = load_test_data(result["data_path"])
    assert len(df) == 10                                    # table continues after the page break
    assert df["eps1"].iloc[-1] == pytest.approx(22.0)       # fraction -> percent
    assert df["q"].iloc[-1] == pytest.approx(384.0)         # MPa -> kPa
    assert df["epsV"].iloc[-1] == pytest.approx(-0.69)      # dilation positive -> contraction positive
    assert "10 points" in result["data_note"]


def test_parse_report_extracts_curves_to_json(stub_llm):
    _check_extracted(tools.parse_report.func(REPORT))
    assert "[0] header: 'Axial strain" in stub_llm["prompt"]  # the LLM sees the tables it must map


def test_parse_report_reads_a_pdf(stub_llm, tmp_path):
    try:
        import pypdf  # noqa: F401
    except ImportError:
        if not shutil.which("pdftotext"):
            pytest.skip("needs pypdf or pdftotext")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.95, REPORT, family="monospace", fontsize=9, va="top")
    fig.savefig(tmp_path / "tx9.pdf")
    plt.close(fig)

    result = tools.parse_report.func(str(tmp_path / "tx9.pdf"))
    _check_extracted(result)
    assert Path(result["data_path"]).name == "tx9_data.json"


def test_parse_report_without_table_asks_for_data(stub_llm):
    result = tools.parse_report.func("CD test at 200 kPa, e0 = 0.68. Curves are shown in Figure 3.")
    assert result["data_path"] is None and "Ask the human" in result["data_note"]


def test_write_config_appends_tests_and_covers_measured_strain(stub_llm, tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "TEMPLATE_CONFIG_PATH", ROOT / "src/calibration/configs/config.yaml")
    parsed = tools.parse_report.func(REPORT)
    first = tools.write_config.func(json.dumps([{k: parsed[k] for k in ("drainage", "sigM0", "e0")}]),
                                    json.dumps([parsed["data_path"]]), round_number=1)
    second = tools.write_config.func(json.dumps([{"drainage": "undrained", "sigM0": 100.0, "e0": 0.75}]),
                                     json.dumps([parsed["data_path"]]), round_number=2,
                                     calibrated_params_json='{"Mtc": 1.25}', previous_config_path=first)
    with open(second) as f:
        cfg = yaml.safe_load(f)
    ns, opt = cfg["NorSandConfig"], cfg["OptimizationSettings"]
    assert ns["simulation"]["drainage"] == ["drained", "undrained"] and ns["simulation"]["batch_size"] == 2
    assert ns["material_parameters"]["soil_state"]["sigM0"] == [200.0, 100.0]
    assert ns["material_parameters"]["plasticity"]["Mtc"] == 1.25
    assert ns["simulation"]["MaxStrainForSim"] == pytest.approx(0.275)  # 22 % measured, with margin
    assert opt["loss_keys"] == ["q", "ue", "epsV"] and len(opt["data_path"]) == 2
    solvers.LBFGSBSolver(cfg, torch.device("cpu"))  # the written config loads


@pytest.mark.slow
def test_reproduce_paper_case_1():
    """Replays the recorded agent decisions of case 1 and checks Tables 4-5 entries (about 15 min)."""
    done = subprocess.run([sys.executable, str(ROOT / "reproduce.py"), "--cases", "1", "--jobs", "1"],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
