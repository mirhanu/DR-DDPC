# Distributionally Robust Data-Driven Predictive Control for Stochastic LTI Systems

This repository contains the implementation of the paper **“Distributionally Robust Data-Driven Predictive Control for Stochastic LTI Systems.”**

It implements **DR-SPC** and **DR-DDPC** controllers and reproduces all numerical experiments in the paper.

---

## 📂 Project Structure

```
DR-DDPC/
│
├── run_all.py                 # Run all experiments
│
├── src/                       # Core implementation
│   ├── controllers/           # DR-SPC, DR-DDPC, baselines
│   ├── mc_runner.py           # Monte Carlo simulation loop
│   ├── experiment_runners.py  # Experiment-specific runners
│   ├── experiments.py         # Experiment definitions
│   ├── cost_functions.py     # Cost and constraint definitions
│   └── util.py               # Plotting, logging, saving utilities
│
├── config/                    # Configuration files
│   └── config.py             # System and simulation parameters
│
├── results/                   # Saved experiment outputs (ignored by git)
│
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## ⚙️ Requirements

Install dependencies:

```
pip install -r requirements.txt
```

---

## 🚀 Running Experiments

Run all experiments:

```
python run_all.py
```

Run selected experiments:

```
python run_all.py --experiments constraint_sweep noise_sweep_zero_mean
```

---

## 📊 Experiments

| Name | Description |
|------|-------------|
| `constraint_sweep` | Constraint violation rate vs. ε_const and β for output constraint \(y \in [0,2]\) |
| `noise_sweep_zero_mean` | Tracking cost vs. disturbance variance (zero-mean noise) |
| `noise_sweep_biased` | Same experiment with biased noise (mean = 0.05) |
| `cost_l1` | L1 tracking cost across noise levels |
| `cost_asymmetric` | Asymmetric quadratic cost across noise levels |

---

## 📁 Results

All results are saved under:

```
results/<experiment_name>_<timestamp>/
```

---

## 📄 Notes

- System parameters and simulation settings are defined in `config/config.py`
- Each experiment is modular and can be extended in `src/experiment_runners.py`
- Controllers are implemented in `src/controllers/`

---

## 📦 Requirements

See `requirements.txt` for full dependencies.

---

## 📜 Citation

If you use this code, please cite the accompanying paper:

---

## ✉️ Contact

For questions regarding the code or paper, please contact the authors.