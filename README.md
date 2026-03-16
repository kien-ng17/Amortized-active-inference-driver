# Active Inference for Autonomous Driving

Source code for the paper **"Active Inference for Autonomous Driving"**.

## Setup

```bash
pip install numpy opencv-python matplotlib pillow gymnasium stable-baselines3 sb3-contrib scipy pandas
```

## Notebooks

### `01_pedestrian.ipynb` — Pedestrian counter-agent
Trains the pedestrian QR-DQN model (`models/ped_model`) that acts as the reactive pedestrian in the crossing environments. Run this first.

### `02_main.ipynb` — Ego-vehicle policies and evaluation
Trains three ego-vehicle SAC policies (Full AIF, No-Epistemic, Mode-of-Belief) and produces all paper figures. Model loading and all plot cells run without re-training — skip the cells marked **SKIP** if trained models are already in `models/`.

| Cell | Description |
|---|---|
| Load pedestrian model | Load `ped_model` into the crossing environments |
| Sanity check | Quick random-policy episode |
| Particle Planner | Tree-search baseline |
| Train all_model | Full AIF policy — **SKIP** |
| Train all_model_NoE | No-epistemic policy — **SKIP** |
| Train all_model_mode | Mode-of-belief policy — **SKIP** |
| Training curves | Plot reward curves from `sb3_logs/` |
| Load models | Load all three trained policies |
| Velocity & belief plot | Single-episode velocity/belief figure |
| False-belief plot | Prior ≠ ground truth scenario |
| Prior sweep — reward | NoE vs AIF cumulative reward |
| Prior sweep — velocity | Mode-of-belief vs AIF velocity |
| IROS trajectory | Snapshot subplots + belief/speed panel |
| Epistemic comparison | With vs without epistemic value |

## Source modules (`src/`)

| File | Description |
|---|---|
| `motion_models.py` | Pedestrian motion model and forward simulation |
| `particle_filter.py` | Particle filters (pragmatic-only and full EFE) and tree-search planner |
| `environments.py` | Gymnasium environments for pedestrian and ego-vehicle agents |
| `visualization.py` | All paper figure functions |
