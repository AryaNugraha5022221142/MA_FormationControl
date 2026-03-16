# Multi-Agent UAV Formation Control with Obstacle Avoidance
A MATLAB simulation of multi-UAV swarm coordination using **Consensus Protocols** and **Artificial Potential Fields (APF)** for formation control and obstacle avoidance

## Overview

This project implements a distributed control framework for a swarm of **6 UAVs** coordinating to:
- Maintain a **regular hexagonal formation** (5 m radius) while tracking a moving goal
- **Avoid static circular obstacles** using Artificial Potential Fields
- **Prevent inter-agent collisions** using cosine-weighted repulsion forces
- **Preserve communication connectivity** throughout the mission

## Key Features

- **Consensus-based formation control** using Graph Laplacian and relative position constraints
- **Artificial Potential Fields** with attractive (goal-seeking) and repulsive (obstacle-avoidance) forces
- **Lyapunov stability analysis** proving asymptotic convergence
- **Distance-weighted adjacency matrix** with smooth cosine decay for communication topology
- **Three mission scenarios**: 60s/3 obstacles, 120s/5 obstacles, 180s/7 obstacles
