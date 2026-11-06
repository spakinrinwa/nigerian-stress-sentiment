"""Recompute every statistic, table and figure of the paper from the restricted data folder.

    python run_all.py            # everything (Experiment III and its controls take about 30 minutes on a laptop CPU)
    python run_all.py --skip-exp3

Inputs: data/restricted/ (see data/README.md), or the folder named by the DATA_DIR environment variable.
Outputs: outputs/ (or OUT_DIR), with each script's printed results in outputs/logs/.
"""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA_DIR", os.path.join(HERE, "data", "restricted"))
OUT = os.environ.get("OUT_DIR", os.path.join(HERE, "outputs"))
S = lambda f: os.path.join(HERE, "scripts", f)
ANN_RAW = os.path.join(DATA, "Human_Sentiment_Annotation_3741.csv")
ANN = os.path.join(OUT, "Human_Sentiment_Annotation_3741_harmonised.csv")
PRED = os.path.join(OUT, "predictions.csv")
OOF = os.path.join(OUT, "experiment3_results", "experiment3_oof_predictions.csv")
STEPS = [
    ("01_harmonise_labels.py", [ANN_RAW]),
    ("02_build_predictions.py", []),
    ("03_experiments_1_2.py", []),
    ("04_gloss_and_macroF1_intervals.py", []),
    ("05_robustness_checks.py", []),
    ("06_reference_symmetry_tests.py", [ANN]),
    ("07_model_symmetry_tests.py", [PRED]),
    ("08_experiment3_supervised.py", [ANN, os.path.join(OUT, "experiment3_results")]),
    ("09_experiment3_controls.py", [ANN, PRED, OOF]),
    ("10_build_tables.py", []),
    ("11_figures_2_3_5_A1.py", [PRED]),
    ("12_figure_4.py", [PRED]),
    ("13_figures_6_7.py", []),
    ("14_graphical_abstract.py", []),
]
if __name__ == "__main__":
    skip = {"08_experiment3_supervised.py", "09_experiment3_controls.py"} if "--skip-exp3" in sys.argv else set()
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True); os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    env = dict(os.environ, DATA_DIR=DATA, OUT_DIR=OUT, MPLBACKEND="Agg")
    for script, args in STEPS:
        if script in skip:
            continue
        print("running", script, flush=True)
        log = os.path.join(OUT, "logs", script.replace(".py", ".txt"))
        with open(log, "w") as fh:
            r = subprocess.run([sys.executable, S(script), *args], cwd=OUT, env=env, stdout=fh, stderr=subprocess.STDOUT)
        if r.returncode:
            sys.exit(f"{script} failed; see {log}")
    print("done; results in", OUT)
