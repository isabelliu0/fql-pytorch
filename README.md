# Flow Q-Learning (FQL) - PyTorch Implementation

This repository contains a PyTorch implementation of Flow Q-Learning (FQL), a method for offline reinforcement learning that leverages flow matching for policy representation. The original paper and JAX implementation can be found [here](https://arxiv.org/abs/2502.02538).

## Overview

Flow Q-Learning (FQL) is a simple and performant offline reinforcement learning method that leverages an expressive flow-matching policy to model arbitrarily complex action distributions in data. The key insight is to train a separate one-step policy that maximizes values while distilling from a flow model trained with behavioral cloning.

This implementation follows the original JAX code structure but uses PyTorch as the backend.

## Installation

```bash
# Install dependencies
pip install torch numpy gymnasium d4rl ogbench wandb ml_collections tqdm pillow
```

## Usage

You can run the FQL agent on different environments using the provided command-line interface:

```bash
# Train an FQL agent on the cube-double-play-singletask-v0 environment
python main.py \
    --env_name cube-double-play-singletask-v0 \
    --run_group FQL-Cube-Double \
    --seed 0 \
    --offline_steps 1000000 \
    --alpha 300

# Train on a visual environment with image observations
python main.py \
    --env_name visual-cube-double-play-singletask-task1-v0 \
    --run_group FQL-Visual-Cube-Double \
    --seed 0 \
    --offline_steps 500000 \
    --alpha 300 \
    --encoder impala_small \
    --frame_stack 3

# Online fine-tuning after offline training
python main.py \
    --env_name antmaze-medium-play-v2 \
    --run_group FQL-Antmaze-Medium-Finetune \
    --seed 0 \
    --offline_steps 500000 \
    --online_steps 1000000 \
    --alpha 10 \
    --balanced_sampling 1
```

## Key Parameters

- `--alpha`: BC coefficient (needs to be tuned for each environment)
- `--flow_steps`: Number of flow steps for the Euler method (default: 10)
- `--encoder`: Visual encoder name for image observations (None, 'impala_small', etc.)
- `--frame_stack`: Number of frames to stack for frame-stacking (useful for visual environments)
- `--balanced_sampling`: Whether to use balanced sampling for online fine-tuning (0 or 1)

## Results

The FQL agent is expected to achieve good performance across a variety of environments, including:
- OGBench manipulation tasks (cube-single, cube-double, scene, puzzle)
- OGBench navigation tasks (antmaze, humanoidmaze, antsoccer)
- D4RL benchmarks (antmaze, adroit)
- Visual environments

## Citation

If you use this code in your research, please cite the original paper:

```
@article{park2025flow,
  title={Flow Q-Learning},
  author={Park, Seohong and Li, Qiyang and Levine, Sergey},
  journal={arXiv preprint arXiv:2502.02538},
  year={2025}
}
```