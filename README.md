# Multi-Agent Pickup and Delivery Solver

A multi-agent pickup and delivery (MAPD) solver implementing windowed priority based search algorithm, hungarian matching, and FIFO pick-up constraints for warehouse automation scenarios.

## Overview

This project provides a solution for coordinating multiple Autonomous Guided Vehicles (AGVs) in warehouse environments, handling AGV kinematic costs, task assignments, collision-avoidance, and FIFO order task pick-up constraints.

## Features

- **Collision-Free Path Planning**: Implements Priority-Based Search (PBS) with windowed planning horizon
- **Conflict Resolution**: Conflict detection and resolution between agents
- **Windowed Collision Resolution**: Implements Rolling-Horizon Collision Resolution (RHCR) to increase computation efficiency
- **Orientation-Aware Pathfinding**: Considers orientation and turning actions of agents during pathfinding
- **FIFO Constraint Handling**: Maintains First-In-First-Out order for task pick-up at each location
- **Hungarian Task Assignment**: Efficient allocation of pickup and delivery tasks to multiple AGVs
- **AGV Trajectory Visualization**: Animated visualization of AGV movements and task execution

## Project Structure

```
multi-agent-pickup-delivery-solver/
├── README.md                # This file
├── Report.pdf               # Detailed technical report
├── solver.py                # Main solver implementation
├── visualizer.py            # Visualization tool for trajectories
├── input/
│   ├── map_data.csv         # Warehouse layout and AGV configuration
│   └── task_csv.csv         # Task definitions (pickup/delivery requests)
└── output/
    ├── agv_trajectory.csv   # Generated AGV trajectories
    └── agv_simulation.mp4   # Animated visualization of solution
```

## Installation

### Prerequisites

For main MAPD solver support:
```bash
# No external libraries needed (tested working on Python 3.10)
```

For visualization video generation support:
```bash
# Python 3.7+ required
pip install pandas matplotlib numpy

# Install FFmpeg (required for MP4 export)
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

## Usage

### Running the Solver

```bash
python solver.py
```

The solver will:
1. Read warehouse configuration from `input/map_data.csv`
2. Load tasks from `input/task_csv.csv`
3. Generate optimal AGV trajectories
4. Output results to `output/agv_trajectory.csv`

### Visualizing Results

```bash
python visualizer.py
```

This will:
1. Load the generated trajectories
2. Create an animated visualization
3. Save the animation as `output/agv_simulation.mp4`

## Input Format

### map_data.csv
Defines the warehouse layout and initial AGV positions:
- Start points (pickup locations)
- End points (delivery locations)
- AGV initial positions and orientations

### task_csv.csv
Specifies pickup and delivery tasks:
- Task ID
- Pickup location
- Delivery location
- Priority level
- Time constraints

## Output Format

### agv_trajectory.csv
Contains timestamped AGV status:
- Timestamp
- AGV name
- Position and orientation
- Loaded status
- Task-ID

## Configuration

Key parameters in `solver.py`:

```python
WINDOW_SIZE = 30     # collisions need to be resolved only within this window of future timesteps
REPLAN_PERIOD = 15   # Replanning frequency (the number of timesteps between each periodic path replan)
```

## Performance

The computation time can be adjusted based on the tradeoff between makespan (the number of timesteps it takes for all tasks to be completed) and efficiency.

Adjusting the window size and replan period will affect this tradeoff:
* Window size:
    * Larger value leads to more stable paths (and possibly better makespan)
    * Smaller value leads to less computation overhead and thus better computation efficiency
* Replan period:
    * Larger value leads to less periodic replans and thus better computation efficiency
    * Smaller value leads to more stable paths

Based on my testing, running the solver on a MacBook (M3 Pro) with the sample input files (20x20 warehouse, 100 tasks, 12 AGVs) results in the following runtime:
| WINDOW_SIZE      |REPLAN_PERIOD     | Runtime (seconds)|
|------------------|------------------|------------------|
|30                |15                |560.45            |
|30                |10                |572.02            |
|20                |10                |570.99            |
|15                |10                |491.24            |
|10                |5                 |287.69            |
|5                 |3                 |221.95            |

## References

1. Xu, Qinghong, et al. "Multi-goal multi-agent pickup and delivery." 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2022. [arxiv.org/pdf/2208.01223](https://arxiv.org/pdf/2208.01223)
2. Li, Jiaoyang, et al. "Lifelong multi-agent path finding in large-scale warehouses." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 35. No. 13. 2021. [arxiv.org/pdf/2005.07371](https://arxiv.org/pdf/2005.07371)
