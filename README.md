# MuJoCo Playground

[![Build](https://img.shields.io/github/actions/workflow/status/google-deepmind/mujoco_playground/ci.yml?branch=main)](https://github.com/google-deepmind/mujoco_playground/actions)
[![PyPI version](https://img.shields.io/pypi/v/playground)](https://pypi.org/project/playground/)
![Banner for playground](https://github.com/google-deepmind/mujoco_playground/blob/main/assets/banner.png?raw=true)


## Installation From Source

```bash
conda create -n Doggy python=3.11
conda activate Doggy
```
Install JAX
```bash
pip install -U "jax[cuda12]"
# Test Installation, should print "gpu"
python -c "import jax; print(jax.default_backend())" 
```
Install MuJoCo Playground
```bash
cd mujoco_playground
pip install -e ".[all]"
# Test Installation
python -c "import mujoco_playground"
```
Install rscope for interactive training visualization
```bash
pip install rscope
```

## Running from CLI
### Policy Training
```bash
python learning/train_jax_ppo.py --env_name Go1Handstand --use_wandb=True --num_evals=1000 --num_timesteps=10000000
```

### Policy Evaluation
Generate a video of rendered policy rollout with top tracing camera
```bash
python learning/train_jax_ppo.py --env_name=Go1Handstand --play_only=True --load_checkpoint_path=path/to/checkpoints --camera=top  --num_videos=1 --episode_length=2500
```

### Training Visualization
To interactively view trajectories throughout training, run
```bash
python learning/train_jax_ppo.py --env_name Go1Handstand --rscope_envs 16 --run_evals=False --deterministic_rscope=True
```
In a separate terminal
```bash
python -m rscope
```


## Citation

This repo is adapted from MuJoCo Playground

```bibtex
@misc{mujoco_playground_2025,
  title = {MuJoCo Playground: An open-source framework for GPU-accelerated robot learning and sim-to-real transfer.},
  author = {Zakka, Kevin and Tabanpour, Baruch and Liao, Qiayuan and Haiderbhai, Mustafa and Holt, Samuel and Luo, Jing Yuan and Allshire, Arthur and Frey, Erik and Sreenath, Koushil and Kahrs, Lueder A. and Sferrazza, Carlo and Tassa, Yuval and Abbeel, Pieter},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/google-deepmind/mujoco_playground}
}
```