# Unconstrained optimization for just-in-time defect prediction

This repository contains the experiment code, summary results, plots, and [final report](paper/report.pdf) for a study of four optimization methods on a neural just-in-time software defect predictor. The methods are gradient descent with Armijo backtracking, heavy-ball momentum, Adam, and L-BFGS. The objective is a smooth, class-weighted, regularized binary cross-entropy loss.

The study uses Camel, Ignite, and Flink from ApacheJIT, plus OpenStack and Qt from the JIT Moving Target replication data. It includes 240 tuning runs and 96 reporting runs. L-BFGS reaches the lowest mean final training objective on each task; validation-selected GD leads mean test average precision on four tasks and Adam leads on Flink. In 85 of 96 reporting runs, continuing beyond the selected checkpoint lowers both training loss and test average precision.

## Repository contents

- `src/`, `study_runner.py`: model, solvers, data preparation, and experiment stages.
- `download_data.py`, `download_external_data.py`: dataset retrieval with fixed SHA-256 checks.
- `analyze_study.py`, `check_zero_saddle.py`, `verify_summary_statistics.py`: analysis and numerical checks.
- `results/summary/`: run-level and aggregate scalar results and the saddle certificate.
- `figures/`: report plots generated from saved experiment results.
- `paper/report.pdf`: the final report.

## Reproduce the study

Use Python 3.12 and install the pinned packages in `requirements.txt`. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python download_data.py
python download_external_data.py
python -c "import bootstrap; from src.study_data import main; main()"
python check_zero_saddle.py
python run_full_study.py
python analyze_study.py --run results/study_v2 --report-dir .
```

The complete training pipeline is computationally substantial. `run_full_study.py` runs tuning, main, external, width, and evaluation stages serially. The raw datasets and saved model arrays are generated locally; the repository tracks their provenance and compact scalar results. Plots in `figures/` and tables in the report document the completed study.

## Data sources

- [ApacheJIT](https://github.com/hosseinkshvrz/apachejit) by Keshavarz and Nagappan.
- [JIT Moving Target](https://github.com/software-rebels/JITMovingTarget) by McIntosh and Kamei.

The downloader scripts verify the exact dataset bytes used in this study. The datasets remain under their original authors' terms.

## License

The original code in this repository is released under the [MIT License](LICENSE). Dataset rights remain with their respective authors.
