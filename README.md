# LEAN

**LEAN: Local Energy-Aware Navigation for Skid-Steer Robots via Deep Reinforcement Learning**.

A DRL local planner policy that saves energy! 

## Install

```bash
conda create -n lean python=3.10.12
conda activate lean
conda env config vars set PYTHONNOUSERSITE=1 && conda activate lean
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Run every command from the repository root.

## Demo

LEAN-0.004 follows a frozen RRT* path in the MuJoCo viewer:

```bash
python demo/run_demo.py
```

## Weights

`training/log/<run>/best_model/best_model.zip`, with `run_config.txt`, `evaluations.npz` and `eval_log.txt`:

| Run | Method |
|---|---|
| `vanilla_drl_s{100,200,300}` | Vanilla-DRL (`w_e = 0`) |
| `lean_we{0.002,0.003,0.004}_s{100,200,300}` | LEAN-`w_e` |

The hardware experiments used `lean_we0.004_s300` and `vanilla_drl_s100`.

## Train

```bash
python training/bunker_sac.py --difficulties easy medium hard --energy-weight 0.004 --seed 100
```

## Reproduce the paper

Benchmark results are in `inference/results/global_paired/20260909_113200/` (`episodes.json`, frozen RRT* paths in `_paths/`). Classical baseline intervals come from `20260616_002825/summary.json`.

| Output | Command |
|---|---|
| Tables III, IV | `python inference/seed_stats.py inference/results/global_paired/20260909_113200` |

It writes `seed_stats.json` and `seed_stats.txt` to the run directory.

Rerun the benchmark on the same frozen paths:

```bash
mkdir -p inference/results/global_paired/rerun
cp -r inference/results/global_paired/20260909_113200/_paths inference/results/global_paired/rerun/
python inference/run_global_inference.py --output_dir inference/results/global_paired/rerun --controllers \
  vanilla_drl_s100:SAC vanilla_drl_s200:SAC vanilla_drl_s300:SAC \
  lean_we0.002_s100:SAC lean_we0.002_s200:SAC lean_we0.002_s300:SAC \
  lean_we0.003_s100:SAC lean_we0.003_s200:SAC lean_we0.003_s300:SAC \
  lean_we0.004_s100:SAC lean_we0.004_s200:SAC lean_we0.004_s300:SAC \
  NMPC:NMPC:0.0 NMPC:NMPC:0.05 NMPC:NMPC:0.1 eadwa:EADWA:0.0 eadwa:EADWA:0.0003 eadwa:EADWA:0.0005
```

`inference/freeze_paths.py` rebuilds `_paths/` from any `episodes.json`.

## Citation

Coming soon with the paper.

## License

MIT, covering the code and the released weights.
