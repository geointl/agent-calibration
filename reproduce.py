"""Reproduce the cross-scenario tables of the paper (Tables 4 and 5) without an LLM.

The agent's two decisions per case that involve the LLM (the follow-up test it
proposed and the warm start of Round 2) are replayed from
results/recorded_agent_decisions.json through the same tools the agent calls.
Everything downstream of those decisions is deterministic, so the calibrated
parameters, losses and Fisher eigenvalues must match results/reference_tables.json.

Usage (about 15 min per case on one core; cases run in parallel):
    python reproduce.py                  # all recorded cases
    python reproduce.py --cases 1 --jobs 1
"""

import argparse
import json
import os
import pickle
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "logs" / "reproduce"
DECISIONS = json.loads((ROOT / "results" / "recorded_agent_decisions.json").read_text())
PARAMS = ["Gamma", "lambda", "Mtc", "N", "chi_tc", "H0", "Hy", "G_at_p0"]
RTOL = 1e-6
# The smallest eigenvalue sits ~13 orders of magnitude below the largest, so its
# trailing digits depend on the BLAS build and thread count.
RTOL_EIG = 1e-2


def run_case(case: str) -> None:
    """Replay one case: Round 1, the recorded follow-up test, Round 2."""
    d = DECISIONS[case]
    os.environ["NORSAND_TEMPLATE_CONFIG"] = str(ROOT / d["template"])
    os.environ["NORSAND_AGENT_LOG_DIR"] = str(OUT_DIR / f"case{case}")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(ROOT))
    from src.calibration.agent import tools  # reads the environment set above

    def calibrate(**kwargs):
        cfg = tools.write_config.func(solver="lbfgsb", n_iterations=50, **kwargs)
        summary = tools.run_calibration.func(cfg)
        assert summary.startswith("Calibration completed"), summary
        return cfg, summary

    cfg1, summary = calibrate(test_specs_json=json.dumps([d["round1_test"]]),
                              data_paths_json=json.dumps([str(ROOT / d["data"])]), round_number=1)
    # The agent reads the calibrated values from the tool's printed summary (6 decimals).
    warm = dict(re.findall(r"^  (\w+): (\S+)$", summary, flags=re.M)) if d["warm_start"] else {}
    test2 = d["round2_test"]
    data2 = tools.generate_virtual_test.func(output_name="virtual_test_2", **test2)
    calibrate(test_specs_json=json.dumps([test2]), data_paths_json=json.dumps([data2]),
              calibrated_params_json=json.dumps({k: float(v) for k, v in warm.items()}),
              round_number=2, previous_config_path=cfg1)


def summarize(case: str) -> dict:
    """Collect the table entries of one case from its two round directories."""
    out = {"round1_test": DECISIONS[case]["round1_test"], "round2_test": DECISIONS[case]["round2_test"]}
    for r in (1, 2):
        rd = OUT_DIR / f"case{case}" / f"round_{r}"
        cfg = json.loads((rd / "calibrated_params.json").read_text())
        materials = {k: v for sec in cfg["NorSandConfig"]["material_parameters"].values() for k, v in sec.items()}
        truth = {k: v["ground_truth"] for sec in cfg["OptimizationSettings"]["parameters"].values()
                 for k, v in sec.items()}
        with open(rd / "results.pkl", "rb") as f:
            history = pickle.load(f)
        with open(rd / "identifiability_diagnostics.pkl", "rb") as f:
            eig = float(pickle.load(f)["eigenvalues"][0])
        out[f"round{r}"] = {"params": {p: materials[p] for p in PARAMS}, "loss": history[max(history)]["data_loss"],
                            "min_eigenvalue": eig}
    out["truth"] = {p: truth[p] for p in PARAMS}
    return out


def tables(results: dict) -> str:
    """Markdown versions of Table 4 (parameter errors) and Table 5 (test design)."""
    def test(t):
        return f"C{t['drainage'][0].upper()}, {t['sigM0']:g} kPa, e0={t['e0']:.3f}"

    t4 = ["| Case | Metric | " + " | ".join(PARAMS) + " |", "|---|---|" + "---:|" * len(PARAMS)]
    t5 = ["| Case | Initial test | Agent-suggested test in R2 | Loss R1 | Loss R2 | eig R1 | eig R2 | Change factor |",
          "|---|---|---|---:|---:|---:|---:|---:|"]
    for case, res in results.items():
        t4.append(f"| {case} | Ref. | " + " | ".join(f"{res['truth'][p]:g}" for p in PARAMS) + " |")
        for r in (1, 2):
            err = [abs(res[f"round{r}"]["params"][p] - res["truth"][p]) / res["truth"][p] * 100 for p in PARAMS]
            t4.append(f"| | R{r} err. (%) | " + " | ".join(f"{e:.2f}" for e in err) + " |")
        r1, r2 = res["round1"], res["round2"]
        t5.append(f"| {case} | {test(res['round1_test'])} | {test(res['round2_test'])} | {r1['loss']:.2e} | "
                  f"{r2['loss']:.2e} | {r1['min_eigenvalue']:.2e} | {r2['min_eigenvalue']:.2e} | "
                  f"{r2['min_eigenvalue'] / r1['min_eigenvalue']:.3g} |")
    return ("Table 4. Relative error (%) of the calibrated parameters\n\n" + "\n".join(t4)
            + "\n\nTable 5. Test design, final loss and smallest Fisher eigenvalue\n\n" + "\n".join(t5) + "\n")


def mismatches(results: dict, reference: dict) -> list[str]:
    """Entries that differ from the reference by more than the relative tolerance."""
    bad = []
    for case, res in results.items():
        for r in ("round1", "round2"):
            got = {**res[r]["params"], "loss": res[r]["loss"], "min_eigenvalue": res[r]["min_eigenvalue"]}
            ref = {**reference[case][r]["params"], "loss": reference[case][r]["loss"],
                   "min_eigenvalue": reference[case][r]["min_eigenvalue"]}
            bad += [f"case {case} {r} {k}: {got[k]:.10g} vs reference {ref[k]:.10g}"
                    for k in got if abs(got[k] - ref[k]) > (RTOL_EIG if k == "min_eigenvalue" else RTOL) * abs(ref[k])]
    return bad


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="+", default=list(DECISIONS), choices=list(DECISIONS))
    parser.add_argument("--jobs", type=int, default=4, help="cases run in parallel")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return run_case(args.worker)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def launch(case):
        with open(OUT_DIR / f"case{case}.log", "w") as log:
            return subprocess.run([sys.executable, __file__, "--worker", case], stdout=log,
                                  stderr=subprocess.STDOUT).returncode

    print(f"Running cases {args.cases} (logs in {OUT_DIR}) ...")
    with ThreadPoolExecutor(args.jobs) as pool:
        codes = list(pool.map(launch, args.cases))
    failed = [c for c, code in zip(args.cases, codes) if code]
    if failed:
        sys.exit(f"Cases {failed} failed; see {OUT_DIR}/case<N>.log")

    results = {case: summarize(case) for case in args.cases}
    (OUT_DIR / "tables.json").write_text(json.dumps(results, indent=1))
    text = tables(results)
    (OUT_DIR / "tables.md").write_text(text)
    print(text)

    reference = json.loads((ROOT / "results" / "reference_tables.json").read_text())
    bad = mismatches(results, reference)
    if bad:
        sys.exit("MISMATCH with results/reference_tables.json:\n  " + "\n  ".join(bad))
    print(f"All values match results/reference_tables.json (relative tolerance {RTOL:g}; "
          f"{RTOL_EIG:g} for the smallest eigenvalue).")


if __name__ == "__main__":
    main()
