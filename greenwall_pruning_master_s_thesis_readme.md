# Design and Implementation of an Autonomous Robotic System for Pruning Indoor Green Walls

[![ ROS 2 ](https://img.shields.io/badge/ROS%202-Humble-blue.svg)](https://docs.ros.org/en/humble/index.html)
[![ Python ](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)
[![ License ](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An autonomous robotic system developed as a Master's Thesis project at the University of Southern Denmark (SDU) to automate the periodic maintenance of multi-floor indoor green walls. The system utilizes a 2D Cable-Driven Parallel Robot (CDPR) as a mobile platform, a hybrid vision pipeline combining classical statistics and deep learning (SAM 2) for target leaf perception, and a dedicated end-effector equipped with a pick-and-pull gripper mechanism.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Mobile Platform: Cable Driven Parallel Robot](#mobile-platform-cable-driven-parallel-robot-cdpr)
- [Plant Perception & Leaf Detection](#plant-perception--leaf-detection)
- [Leaf Pruning Manipulator](#leaf-pruning-manipulator)
- [System Integration & Execution](#system-integration--execution)
- [Results & System Validation](#results--system-validation)
- [Future Work](#future-work)
- [Setup & Running Code](#setup--running-code)

---

## Project Overview
Indoor vertical green walls improve indoor air quality, reduce cooling requirements, and boost occupant psychological well-being. However, maintaining large multi-floor plant installations requires periodic pruning of withered leaves. Accessing these vertical surfaces manually presents safety challenges (such as reaching across bridge guard rails) and high labor demands.

<img width="800" alt="canvas" src="https://github.com/user-attachments/assets/dc0d213c-7910-47c6-871f-64d24838b726" />



This project proposes an autonomous pruning robot designed for complex vertical indoor spaces. As a proof of concept, a $1\text{ m} \times 1\text{ m}$ planar prototype was developed to execute continuous leaf detection and autonomous removal sequence over plant foliage.

---

## Mobile Platform: Cable Driven Parallel Robot (CDPR)
To maintain a minimal structural footprint without blocking natural daylight or altering building aesthetics, a planar Cable-Driven Parallel Robot (CDPR) is chosen as the mobile platform.

<img height="450" alt="image" align=center src="https://github.com/user-attachments/assets/612459eb-e3d2-4f26-8a76-a7a8d9e8cae8" />

### Kinematics and Inverse Kinematics
The CDPR operates across 3 Degrees of Freedom (DoF): planar translations $p = [x, y]^T$ and dynamic platform rotation $\theta$. The unique vector transformation for the length of cable $i$ is formulated as:

$$l_i = \| p - b_i + R_E^G q_i \|$$

where:
- $b_i$ denotes fixed frame cable anchor coordinates.
- $q_i$ is the fixed displacement vector of anchor point $i$ on the moving platform.
- $R_E^G$ is the $2\text{D}$ rotation matrix translating platform relative dynamics to global coordinates.

<img height="350" alt="image" src="https://github.com/user-attachments/assets/ba626645-e1cb-4274-8555-45d17061a729" />


### Forward Kinematics
Because cable-driven systems do not admit closed-form analytic solutions for forward kinematics, position estimation is solved numerically by minimizing the Mean Squared Error (MSE) of cable residuals:

$$\mathbf{x}_{\text{present}} = \arg \min_{\mathbf{p}, \theta} \frac{1}{4} \sum_{i=1}^{4} \left( l_i - \|\tilde{l}_i(\mathbf{p}, \theta)\| \right)^2$$

Sequential Least Squares Programming (SLSQP) is used to update the position real-time, seeded by $x_{t-1}$.

### Control Approaches & Trade-offs
**Kinematic Velocity Control**: Combines a linear trajectory generator with closed-loop cable correction:

$$\dot{l}_{\text{desired}} = \dot{l}_{\text{traj}} + K_p(l_{\text{ideal}} - l_{\text{actual}})$$

*Performance:* Achieved sub-centimeter positional accuracy across **98.7%** of the workspace.

**Force-Based Dynamic Control**: Evaluated using a Torque Distribution Algorithm utilizing the structure matrix null space $S \cdot \mathbf{t} = \mathbf{u}$:

$$\mathbf{t}_{\text{desired}} = \mathbf{t}^\dagger + \lambda \mathbf{n}$$

*Outcome:* Significant internal gear friction within high-reduction servo motors (272.5:1 ratio) led to asymmetric forces, limiting the accurate workspace area to 13.5%. Kinematic velocity control was selected for production integration.

---

## Plant Perception & Leaf Detection

Target identification requires isolating withered yellow leaves from vibrant background foliage under fluctuating natural lighting conditions.

<img height="450" alt="image" src="https://github.com/user-attachments/assets/c369613c-951d-4324-ba9c-7a32624658e4" />



### Mahalanobis-Prompted SAM 2 Pipeline
To balance processing speed with high geometric fidelity, a hybrid zero-shot vision architecture was developed:

1. **Statistical Pre-filtering**: Pass raw camera frames through Mahalanobis Distance thresholding in CieLAB color space to obtain sample covariance distributions ($\mathbf{\hat{\Sigma}}$):
   $$D_i = \sqrt{(\mathbf{c}_i - \mathbf{c}_t)^T \mathbf{\Sigma}^{-1} (\mathbf{c}_i - \mathbf{c}_t)}$$
2. **Foundational Prompting**: Extract Euclidean distance maxima on the pre-filtered candidate region to derive spatial seed coordinates.
3. **Instance Masking**: Supply seed coordinates to Meta’s Segment Anything Model 2 (SAM 2) to obtain clean edge contours without domain fine-tuning.
4. **Color Verification**: Compute color percentiles on isolated mask bounds to discard false positive predictions.

### Spatial Reconstruction
Pixel mask centroids $(u_i, v_i)$ are converted to global 3D coordinates using camera intrinsic parameters $K$ derived from pinhole projection, coupled with depth maps $Z_i$:

$$x_i^C = \frac{(u_i - c_x) \cdot Z_i}{f_x}, \quad y_i^C = \frac{(v_i - c_y) \cdot Z_i}{f_y}$$

Point clouds are sequentially transformed through end-effector frame $\{E\}$ into global system coordinates $\{G\}$.

---

## Leaf Pruning Manipulator

Executing leaf removal requires a compact, targeted mechanical system mounted directly on the CDPR end-effector platform.

<img height="250" alt="image" src="https://github.com/user-attachments/assets/0ab2e236-fb35-4a6f-9abb-c35285865770" />
<img height="250" alt="image" src="https://github.com/user-attachments/assets/a5fd6b90-c0b7-47a8-bced-042be668b0d3" />
<img height="250" alt="image" src="https://github.com/user-attachments/assets/28acb066-7e4d-4685-9da5-604d4ee9a7aa" />



### Mechanical Design & Strategy
- **Pruning Method**: A pick-and-pull grasping technique is utilized to strip leaves from nodes without requiring bulky high-torque rotary blades.
- **Gripper Kinematics**: A dual-finger gear-driven mechanism optimized to squeeze leaf stems firmly before executing a controlled backward pull displacement using the CDPR motion platform.
- **Form Factor**: Designed under strict operational envelopes: total end-effector thickness is under $30\text{ cm}$ to navigate guard rails and narrow structural clearances safely.

---

## System Integration & Execution

The system integrates path planning, vision processing, and end-effector motion into an automated loop:

<img height="280" alt="image" src="https://github.com/user-attachments/assets/e454402d-0db3-4abe-90bd-0f27a489d994" />


1. **Coverage Raster Scan**: The CDPR traverses the wall face following optimized overlapping horizontal scanning rows.
2. **Visual Detection Trigger**: On-the-fly camera inference captures yellow leaves, halting platform movement.
3. **Approach Trajectory**: Inverse kinematics direct the platform TCP toward the leaf target vector.
4. **Grasping & Pruning Cycle**: The manipulator clamps the stem, executes a pulling movement, drops the leaf, and returns the platform to the scanning sequence.

---

## Results & System Validation

The fully integrated autonomous system was evaluated in realistic experimental trials comprising **101 total pruning attempts** over **85 distinct target leaves**.

### Performance Metrics
- **Single-Attempt Success Rate**: **70.3%**
- **Cumulative Retry Success Rate**: **83.5%**
- **CDPR Kinematic Coverage**: **98.7%** sub-centimeter workspace tracking accuracy.
- **Perception Segmentation Quality (IoU)**: **79.4%** achieved via prompted SAM 2, down from an 8.26-second unprompted inference latency to interactive speeds.

### Key Analysis & Findings
- **Forward Kinematic Errors**: Statistical analysis confirmed accuracy changes proportional to target depth from the image plane. Cable pulley anchor point displacement ($b_i$ drift) under load represents the main structural source of position error near workspace boundaries.
- **Drive Gear Friction**: Dynamixel current-based dynamic force control requires direct-drive setups; friction forces heavily distort unconstrained torque control.

---

## Future Work

- **Dynamic Cable Modeling**: Incorporating variable exit points ($b_i$) on corner pulleys within the kinematic loop to eliminate position drift at extreme edges.
- **Visual Feedback Servo Control**: Implementing direct visual tracking inside the closed loop to adjust grasping trajectories dynamically prior to stem engagement.
- **High-Torque Direct Drive Actuation**: Upgrading winch hardware to low-friction motors to enable continuous dynamic force control and tension monitoring.
- **Scaling to Building Scale**: Expanding system structural frames to span multi-floor plant installations ($9\text{ m} \times 15\text{ m}$).

---

## Setup & Running Code

### Prerequisites
- ROS 2 Humble
- Python 3.10+
- CUDA-compatible GPU (recommended for SAM 2 inference)

### Installation

1. Clone the repository into your ROS 2 workspace:
```bash
cd ~/ros2_ws/src
git clone https://github.com/alexellegaard/Greenwall-Pruning.git
```

2. Install dependencies:
```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
pip install -r src/Greenwall-Pruning/requirements.txt
```

3. Build the workspace:
```bash
colcon build --symlink-install
source install/setup.bash
```

### Running the System

1. Launch the CDPR controller hardware interface:
```bash
ros2 launch cdpr_control cdpr_control.launch.py
```

2. Start the leaf detection perception node:
```bash
ros2 launch leaf_detection perception.launch.py
```

3. Execute the autonomous pruning state machine:
```bash
ros2 run pruning_runner system_integration_node
```
