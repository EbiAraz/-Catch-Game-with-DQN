# Catch Game With DQN

A clean Deep Q-Network project where an agent learns to catch a falling ball in a custom Gymnasium environment.

## Repository

Project link: https://github.com/EbiAraz/-Catch-Game-with-DQN
Run script link: https://github.com/EbiAraz/-Catch-Game-with-DQN/blob/main/Catch%20Game(DQN).py

## Features

- Custom Catch environment built with Gymnasium API
- DQN with replay memory and target network
- Fast training path with TensorFlow GradientTape
- Optional Pygame rendering for test episodes
- Saved plots for training and test performance

## Project Files

```text
.
├── Catch Game(DQN).py
├── requirements.txt
├── model_weights.weights.h5      # created after training
├── results.png                   # created after training
└── test_results.png              # created after test
```

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python "Catch Game(DQN).py"
```

## Run Options

When the script starts, it shows interactive options such as:

- Train from scratch
- Watch trained gameplay
- Watch with continuous learning
- Test and plot results

## DQN Flow

1. Reset environment and collect transitions.
2. Store transitions in replay memory.
3. Sample mini-batches and compute Q targets.
4. Update policy network and periodically sync target network.
5. Save model weights and evaluation plots.

## Expected Outputs

- model_weights.weights.h5
- results.png
- test_results.png

## Notes

- GPU acceleration is used automatically when TensorFlow detects a compatible GPU.
- On native Windows with modern TensorFlow builds, CUDA support can require WSL2 or DirectML.
- For graphical test rendering, install and enable pygame.
