# AdaptiveStream-RL

A research-oriented prototype for **Adaptive Video Streaming** using
**Reinforcement Learning**.

------------------------------------------------------------------------

## Overview

Adaptive video streaming is a technique that dynamically adjusts video
quality according to current network conditions to improve the user's
**Quality of Experience (QoE)**.

This project implements a simplified Adaptive Bitrate (ABR) streaming
simulator where a Reinforcement Learning agent learns to select the most
appropriate video bitrate under varying network bandwidth conditions.

The project was developed as a self-learning research prototype while
preparing for PhD applications in Artificial Intelligence and
Intelligent Networking.

------------------------------------------------------------------------

## Features

-   Adaptive streaming environment
-   Dynamic bandwidth simulation
-   Playback buffer simulation
-   QoE-based reward function
-   Rule-based baseline agent
-   Q-Learning prototype
-   DQN training pipeline using Stable-Baselines3
-   Gymnasium-compatible environment
-   Training and evaluation scripts
-   Result visualization

------------------------------------------------------------------------

## Project Structure

``` text
AdaptiveStream-RL/
│
├── environment/
│   └── streaming_env.py
│
├── agents/
│   └── q_learning_agent.py
│
├── train_dqn.py
├── evaluate.py
│
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

## Reinforcement Learning Environment

### State

-   Current network bandwidth
-   Current playback buffer
-   Previously selected bitrate

### Action

The agent selects one of the following bitrates:

-   240p
-   360p
-   480p
-   720p
-   1080p

### Reward

The reward function is based on Quality of Experience (QoE) and
currently considers:

-   Video quality
-   Playback interruptions (rebuffering)
-   Bitrate switching penalty

------------------------------------------------------------------------

## Reinforcement Learning Pipeline

``` text
State
   ↓
Agent
   ↓
Bitrate Selection
   ↓
Streaming Environment
   ↓
Reward (QoE)
   ↓
Next State
```

------------------------------------------------------------------------

## Installation

Clone the repository:

``` bash
git clone https://github.com/your-username/AdaptiveStream-RL.git
cd AdaptiveStream-RL
```

Create a virtual environment:

``` bash
python -m venv .venv
```

Activate it:

Linux / macOS

``` bash
source .venv/bin/activate
```

Windows

``` bash
.venv\Scripts\activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Training

``` bash
python train_dqn.py
```

------------------------------------------------------------------------

## Evaluation

``` bash
python evaluate.py
```

------------------------------------------------------------------------

## Visualization

``` bash
tensorboard --logdir logs/
```
TensorBoard at http://localhost:6006/

you will see:

- Loss
- Episode reward
- Q-value trends

------------------------------------------------------------------------

## Technologies

-   Python
-   NumPy
-   Gymnasium
-   Stable-Baselines3
-   PyTorch
-   Matplotlib
-   Pandas

------------------------------------------------------------------------

## Current Status

This repository is a **research prototype** intended for experimentation
with reinforcement learning for adaptive video streaming.

Planned future improvements include:

-   Real network traces
-   MPEG-DASH support
-   More realistic QoE models
-   PPO implementation
-   Benchmark comparison

------------------------------------------------------------------------

## Future Work
Extension to multi-agent reinforcement learning
Real-world deployment in network systems
Improved reward engineering
Comparison with advanced RL algorithms (SAC, PPO, DDPG)

------------------------------------------------------------------------
## Author
Fatemeh KHosravi
Background: M.Sc. Mathematical Statistics
Experience: 6 years Software Development (Python, Django, FastAPI)
Current focus: Reinforcement Learning & AI-driven systems
------------------------------------------------------------------------
## License

This project is licensed under the MIT License.
