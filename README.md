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


### Training

This project could be trained with two different implementations of **PPO** algorithm, logs and checkpoints are saved in `logs` directory.

- To train with **[RSL-RL](https://github.com/leggedrobotics/rsl_rl)**

  ```bash
  python learning/train_rsl_rl.py --env_name Go1Handstand --use_wandb=True
  ```

- To train with **[Brax](https://github.com/google/brax)**

  ```bash
  python learning/train_jax_ppo.py --env_name Go1Handstand --use_wandb=True --num_evals=1000 --num_timesteps=10000000
  ```

### Evaluation

Render the behaviour from the resulting policy with top tracing camera

- Render a policy trained with **RSL-RL**

  > **[NOTE]** please make sure the folder of the run to be evaluated is under the directory of `logs/rslrl-training-logs`

  ```bash
  python learning/train_rsl_rl.py --env_name Go1Handstand --play_only --load_run_name <run_name> --camera=top
  ```

- Render a policy trained with **Brax**

  ```bash
  python learning/train_jax_ppo.py --env_name=Go1Handstand --play_only=True --load_checkpoint_path=path/to/run_name/checkpoints --camera=top  --num_videos=1 --episode_length=2500
  ```

where `run_name` could be found at
- `Run name` printed in the real-time console of **RSL-RL**
- `Experiment Name` printed at the begining of the training of **Brax**

### Interactive Visualization

> **[NOTE]** This function is ONLY available for **Brax**

Interactively view trajectories throughout training

```bash
python learning/train_jax_ppo.py --env_name Go1Handstand --rscope_envs 16 --run_evals=False --deterministic_rscope=True
```

Alternatively, you can add `--load_checkpoint_path=...` to evaluate (and keep training) a trained policy

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