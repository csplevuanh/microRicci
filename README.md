# MicroRicci: A Greedy and Local Ricci Flow Solver

**Le Vu Anh**, Dinh Duc Nha Nguyen, Phi Long Nguyen, Keshav Sood

This repository contains the full Python implementation of MicroRicci, including:

- **Core solver**: greedy syndrome-decoding + self-tuning modules
- **Offline training**: small MLPs for vertex selection and step-size regression
- **Experiments**: benchmarks vs. ORC-Pool, GNRF, Learning-Ricci
- **Visualization**: scripts to reproduce all figures in ICML paper

## Installation
```bash
git clone https://github.com/yourusername/microRicci.git
cd microRicci
pip install -r requirements.txt
```

## Usage
```bash
# Preprocess data
python data/preprocess.py --input raw_meshes/ --output data/processed/

# Train selector and regressor
python training/train_selector.py
python training/train_regressor.py

# Run experiments
python experiments/run_microRicci.py --meshes data/processed/ --output results/

# Plot figures
python viz/plot_convergence.py results/convergence.json
python viz/plot_uv_mos.py results/uv_mos.csv
```
