# Autonomous Indoor Greenwall Pruning Robot

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![ROS 2](https://img.shields.io/badge/ROS2-Humble%2FIron-orange)](https://docs.ros.org/)

This repository contains the complete implementation and empirical validation for the Master’s Thesis project **"Design and Implementation of an Autonomous Robotic System for Pruning Indoor Green Walls"**, completed as part of the MSc in Robot Systems (Drones and Autonomous Systems) degree at the **University of Southern Denmark (SDU)** by **Alex Mejlvang Ellegaard** and **Anders Lind-Thomsen**, under the supervision of **Oskar Palinko**.

---

## Demonstrations

<p align="center">
  <img src="videos/Pruning.gif" width="45%" title="Physical System Pruning Sequence"/>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="videos/Vision pipeline.gif" width="45%" title="Perception Pipeline Output"/>
</p>

> **Note:** Extended high-definition video demonstrations of full autonomous runs, leaf detection, and pruning validation experiments are hosted on [Google Drive](https://drive.google.com/file/d/1_ytfkOsZ_eVM8iPWUoouuIQAFR-C1kbh/view?usp=drive_link).

---

## Executive Summary & Motivation

Vertical indoor green walls offer substantial aesthetic, environmental, and psychological benefits in modern architectural spaces. At the Technical Faculty of SDU (Building 44), a multi-floor $9\text{ m} \times 15\text{ m}$ indoor green wall enhances air quality and thermal insulation. However, biological plants undergo constant life cycles, requiring periodic maintenance to remove dead or chlorotic (yellow) leaves.

Human manual pruning is severely restricted due to structural height, tight access corridors, and safety regulations (e.g., union rules forbidding staff from reaching beyond bridge guardrails). While automated irrigation systems are widespread, no complete, fully autonomous physical solution existed for green wall leaf pruning.

```
+-----------------------------------------------------------------------------------+
|                                 SYSTEM ARCHITECTURE                               |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  +------------------------+  +------------------------+  +---------------------+  |
|  |  Subsystem 1: Platform |  |  Subsystem 2: Vision   |  | Subsystem 3: Pruner |  |
|  |                        |  |                        |  |                     |  |
|  |   2D Planar Cable-     |  |   RGB-D Camera         |  |   2-DOF Custom      |  |
|  |   Driven Parallel      |  |   Mahalanobis + SAM 2  |  |   "Grasp-and-Pull"  |  |
|  |   Robot (CDPR)         |  |   Hybrid Pipeline      |  |   Mechanic Gripper  |  |
|  +-----------+------------+  +-----------+------------+  +----------+----------+  |
|              |                           |                          |             |
|              +---------------------------+--------------------------+             |
|                                          |                                        |
|                                          v                                        |
|                          +--------------------------------+                       |
|                          |     Integrated System State    |                       |
|                          |     Machine & Path Planning    |                       |
|                          +--------------------------------+                       |
+-----------------------------------------------------------------------------------+
```

To address this challenge, this project presents the **Pruning Robot**: a modular, end-to-end autonomous robotic platform operating on a 2D Planar **Cable-Driven Parallel Robot (CDPR)** framework. The platform navigates a vertical $1\text{ m}^2$ proof-of-concept workspace, utilizes a novel statistical-AI hybrid computer vision pipeline to isolate targets in real-time under variable lighting, and executes a customized "grasp-and-pull" mechanical pruning routine.

---

## Deep Dive: System Architecture & Subsystems

The system is partitioned into three core structural and computational subsystems, followed by an overarching state machine integration layer.

```
                      +-----------------------------+
                      |      Main ROS 2 Loop        |
                      +--------------+--------------+
                                     |
             +-----------------------+-----------------------+
             |                       |                       |
             v                       v                       v
     +---------------+       +---------------+       +---------------+
     | 1. CDPR Platform|     | 2. Perception |       | 3. Manipulator|
     |   Kinematics  |       |    Pipeline   |       |   Actuation   |
     +---------------+       +---------------+       +---------------+
```

---

### Part 1: Mobile Platform — Cable Driven Parallel Robot (CDPR)

#### 1. System Topology & Kinematic Formulation
A Cable-Driven Parallel Robot utilizes lightweight tensioned lines spooled by corner winches to position an End Effector (EE). For a planar configuration operating in $x, y$ with planar rotation $\theta$, the system possesses $n = 3$ Degrees of Freedom (DoF). Because flexible cables can only exert tensile forces ($t_i \ge 0$), achieving full static constraint requires $m = n + 1 = 4$ independent cables.

Let $\{G\}$ denote the global frame and $\{E\}$ denote the moving end effector frame. The position vector of the end effector center (Tool Center Point - TCP) is $p = [x, y]^\top$, and its orientation is $\theta$. The global base anchor points of the corner pulleys are $b_i \in \mathbb{R}^2$, and the cable connection vectors relative to the end effector center in frame $\{E\}$ are $q_i^E \in \mathbb{R}^2$ for $i \in \{1, 2, 3, 4\}$.

```
b_1 o=================================================o b_2
    | \                                             / |
    |  \                 q_1   q_2                 /  |
    |   \                 o---o                   /   |
    |    \                | E |                  /    |
    |     \               o---o                 /     |
    |      \             q_4   q_3             /      |
    |       \                                 /       |
b_4 o=================================================o b_3
```

The loop-closure equation for the $i$-th cable vector $l_i^G$ in global coordinates is given by:
$$l_i^G = p^G - b_i^G + R_E^G(\theta) q_i^E$$

Where $R_E^G(\theta) \in SO(2)$ is the 2D rotation matrix:
$$R_E^G(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

The scalar cable length $l_i$ is the Euclidean norm:
$$l_i = \|l_i^G\|_2 = \sqrt{(l_{i,x}^G)^2 + (l_{i,y}^G)^2}$$

#### 2. Forward Kinematics (FK)
Because CDPRs are parallel mechanisms, their Inverse Kinematics (IK) are analytical and direct, whereas Forward Kinematics lack a closed-form solution. The current pose $x_{\text{present}} = [x, y, \theta]^\top$ is estimated numerically by minimizing the Mean Squared Error (MSE) between actual measured cable spooled lengths $\tilde{l}_i$ (from motor encoders) and theoretical lengths derived from the IK model:

$$x_{\text{present}} = \arg\min_{x \in \mathbb{R}^2 \times S^1} \sum_{i=1}^{4} \left( \tilde{l}_i - \|p - b_i + R_E^G(\theta) q_i^E\| \right)^2$$

This non-linear optimization is executed at $50\text{ Hz}$ using Sequential Least Squares Programming (SLSQP), seeded with the pose estimate $x_{t-1}$ from the preceding execution cycle to guarantee rapid convergence without local minima entrapment.

#### 3. Statics & Tension Structure Matrix
The net wrench vector $u \in \mathbb{R}^3$ acting on the TCP consists of two Cartesian force components and a rotational torque:
$$u = \begin{bmatrix} F_x & F_y & \tau_z \end{bmatrix}^\top$$

The relationship mapping cable tension forces $t = [t_1, t_2, t_3, t_4]^\top$ to the Cartesian wrench $u$ is governed by the Structure Matrix $S(x) \in \mathbb{R}^{3 \times 4}$:
$$u = S t$$

Where $S$ is composed of normalized cable unit vectors $\hat{l}_i = \frac{l_i^G}{\|l_i^G\|}$ and moment arms:
$$S = \begin{bmatrix} \hat{l}_{1,x} & \hat{l}_{2,x} & \hat{l}_{3,x} & \hat{l}_{4,x} \\ \hat{l}_{1,y} & \hat{l}_{2,y} & \hat{l}_{3,y} & \hat{l}_{4,y} \\ (R_E^G q_1^E \times \hat{l}_1)_z & (R_E^G q_2^E \times \hat{l}_2)_z & (R_E^G q_3^E \times \hat{l}_3)_z & (R_E^G q_4^E \times \hat{l}_4)_z \end{bmatrix}$$

#### 4. Control Implementations & Comparative Evaluation
Two distinct control architectures were built and empirically validated on the physical system:

##### A. Kinematic Trajectory Controller
Uses a feedforward Linear Trajectory Planner coupled with a parallel feedback Cable Correction Controller to regulate motor spooling velocities:
$$\dot{l}_{\text{desired}} = \dot{l}_{\text{traj}} + K_p \left( l_{\text{ideal}} - l_{\text{actual}} \right)$$

This approach decouples Cartesian velocity from workspace distance, eliminating asymptotic deceleration and chattering via deadband thresholds ($\epsilon_p, \epsilon_\theta$).

##### B. Force / Impedance Controller
Maps Cartesian tracking errors to desired wrenches ($u_{\text{desired}}$) using an impedance model, resolving redundant cable tensions via the Moore-Penrose pseudoinverse $S^\dagger$ and a Torque Distribution Algorithm operating in the 1D null space $N(S)$:
$$t_{\text{desired}} = S^\dagger u_{\text{desired}} + \lambda n$$

Where $n \in \text{null}(S)$ and $\lambda \in \mathbb{R}$ is calculated dynamically to enforce $t_i \ge T_{\text{min}} = 5\text{ N}$.

```
                 +-------------------------------------------------------+
                 |                 EXPERIMENTAL RESULTS                  |
                 +-------------------------------------------------------+
                 | Control Type | Metric            | Value              |
                 +--------------+-------------------+--------------------+
                 | Kinematic    | Reachable Area    | 0.7301 m² (99.5%)  |
                 |              | Accurate (<10mm)  | 0.7244 m² (98.7%)  |
                 |              | Median XTE        | 1.8 mm             |
                 +--------------+-------------------+--------------------+
                 | Force        | Reachable Area    | 0.1954 m² (26.6%)  |
                 |              | Accurate (<10mm)  | 0.0989 m² (13.5%)  |
                 |              | Median XTE        | 12.0 mm            |
                 +-------------------------------------------------------+
```

```
           KINEMATIC CONTROL WORKSPACE              FORCE CONTROL WORKSPACE
         +----------------------------+          +----------------------------+
         |############################|          | . . . . . . . . . . . . . .|
         |############################|          | . . . . . +------+ . . . . |
         |############################|          | . . . . . |Accurate| . . . |
         |############################|   vs     | . . . . . +------+ . . . . |
         |############################|          | . . Reachable . . . . . . .|
         |############################|          | . . . . . . . . . . . . . .|
         +----------------------------+          +----------------------------+
          98.7% Sub-centimeter Accuracy            13.5% Sub-centimeter Accuracy
```

**Experimental Insight:** Force control failed due to unmodeled internal mechanical friction in the Dynamixel XM540-W270 actuators (272.5:1 gear reduction ratio). The unmodeled gear box friction destroyed the force balance assumptions, leading to asymmetric tension spikes $>40\text{ N}$ that triggered system safety shutdowns. Conversely, **Kinematic Control achieved sub-centimeter positioning across 98.7% of the theoretical workspace**, making it the selected motion engine.

---

### Part 2: Leaf Detection — Camera-Based Plant Perception

To isolate chlorotic target leaves under variable natural ambient light, a high-speed, real-time perception pipeline was engineered.

```
  Raw RGB-D Frame        Mahalanobis Distance         Morphological Cleanup
+-----------------+     +--------------------+     +------------------------+
|                 |     |                    |     |                        |
|   [ Green Wall  |     |   [ High-Prob      |     |   [ Noise Filtered     |
|     Plants ]    +---->+     Target Seeds ] +---->+     Mask & Centroid ]  |
|                 |     |                    |     |                        |
+-----------------+     +--------------------+     +-----------+------------+
                                                                |
                                                                v
   Target 3D Point      Color Verification         SAM 2 Segmentation
+-----------------+     +--------------------+     +------------------------+
|                 |     |                    |     |                        |
|  [ Pose in {G}  |<----+ [ 15th Percentile  |<----+ [ High-Precision Mask  |
|    Frame ]      |     |   Color Check ]    |     |   Generation ]         |
+-----------------+     +--------------------+     +------------------------+
```

#### 1. Statistical Color Space Segmentation (Mahalanobis Distance)
Standard Euclidean distance in RGB/HSV spaces fails under dynamic lighting because it ignores inter-channel covariance. The system uses the **Mahalanobis Distance** ($D_M$), which measures distance relative to a calibrated Gaussian distribution of yellow leaf colors:

$$D_M(x) = \sqrt{(x - \mu)^\top \Sigma^{-1} (x - \mu)}$$

Where $x \in \mathbb{R}^3$ is a pixel color vector in $LAB$ color space, $\mu \in \mathbb{R}^3$ is the sample mean vector of chlorotic plant matter, and $\Sigma \in \mathbb{R}^{3 \times 3}$ is the covariance matrix. A threshold $d_{\text{thresh}} = 2.5$ isolates target candidate pixels.

#### 2. Seeded Segment Anything Model 2 (SAM 2) Hybrid Pipeline
While Mahalanobis thresholding is extremely fast ($\sim 12\text{ ms}$), its raw masks lack fine geometric edge accuracy. Directly applying Meta's SAM 2 foundational model automatically across the whole frame yields precise contours but requires unacceptable processing time ($\sim 8.26\text{ s}$ per frame on a GTX 1650Ti / Jetson Orin Nano class GPU).

To solve this, we designed a **Hybrid Pipeline**:
1. **Fast Candidate Identification:** The Mahalanobis module computes the candidate mask and applies distance-transform peak detection to calculate candidate centroids $(u_c, v_c)$.
2. **Promoted Visual Prompting:** These centroids serve as positive prompt points fed directly into SAM 2.
3. **Refinement & Verification:** SAM 2 outputs a high-definition boundary mask around the prompt. A final color-verification check evaluates the 15th percentile $LAB$ vector within the SAM 2 mask to discard false-positive prompts (e.g., wooden support structures or bright leaf veins).

```
+-----------------------------------------------------------------------------------------+
|                              PERCEPTION EVALUATION BENCHMARK                            |
+-----------------------------------------------------------------------------------------+
| Pipeline Variant             | Detection Accuracy (%) | Mask Quality (IoU %) | Time (s) |
+------------------------------+------------------------+----------------------+----------+
| Mahalanobis Only             | 68.7%                  | 36.1%                | 0.0120 s |
| SAM 2 Automatic              | 66.0%                  | 79.4%                | 8.2611 s |
| **Combined Hybrid (Ours)**   | **76.6%**              | **87.7%**            | 0.6135 s |
+-----------------------------------------------------------------------------------------+
```

**Statistical Significance:** Friedman non-parametric testing followed by Wilcoxon signed-rank post-hoc analysis confirmed that the Hybrid Pipeline achieves **statistically superior target accuracy ($p = 0.011$) and IoU quality ($p = 0.038$)** relative to standalone methods, while executing **$13.5\times$ faster than full SAM 2**.

#### 3. Spatial Coordinate Transformations (Pinhole & Frames)
Using the Intel RealSense D435 RGB-D sensor, the 2D pixel mask is projected into $3D$ space. For each pixel $(u_i, v_i)$ with registered depth $z_i^C$, the spatial coordinate $p^C = [x^C, y^C, z^C]^\top$ in the Camera Frame $\{C\}$ is:

$$x^C = \frac{(u_i - c_x) \cdot z_i^C}{f_x}, \quad y^C = \frac{(v_i - c_y) \cdot z_i^C}{f_y}$$

Where $f_x, f_y$ and $c_x, c_y$ represent the intrinsic focal lengths and principal point. The 3D point cloud of the target leaf is mapped to global CDPR coordinates $\{G\}$ via rigid frame transformations:

$$p^G = R_E^G(\theta) \left( R_C^E p^C + t_C^E \right) + p_{\text{TCP}}^G$$

Where $(R_C^E, t_C^E)$ represents the extrinsic transformation from camera to end-effector frame.

---

### Part 3: Leaf Pruning — Manipulator Design

```
         GRASP PHASE                           PULL PHASE
     +-----------------+                   +-----------------+
     |   Two-Finger    |                   |   Manipulator   |
     |   Jaws Open     |                   |   Retracts      |
     |                 |                   |                 |
     |    /       \    |                   |     |   |       |
     |   /  LEAF   \   |  == Closes ==>    |     |[L]|       |  == Retracts ==> (Petiole Detaches)
     |  o           o  |                   |     |   |       |
     |                 |                   |     |   |       |
     +-----------------+                   +-----------------+
```

#### 1. "Grasp-and-Pull" Mechanical Principle
Unlike agricultural harvesters that utilize complex rotary blades, shears, or laser cutters, green wall maintenance benefits from avoiding active cutting edges (which dull rapidly, introduce wire safety hazards, and risk damaging neighboring healthy vines).

Botanical testing revealed that plant leaf petioles exhibit weak tensile shear limits when pulled along their normal growth orientation. The end-effector is designed as a custom **Two-Finger Mechanized Gripper** executing a deterministic **Grasp-and-Pull** sequence:
1. **Approach:** End-effector aligns with target leaf center using depth spatial feedback.
2. **Grasp:** Parallel 3D-printed fingers with high-friction silicone pads actuate to pinch the leaf blade.
3. **Pull:** The CDPR platform executes a quick feedforward linear vector displacement away from the vertical plane, severing the petiole cleanly at the stem node without requiring additional arm DoFs.

#### 2. Experimental Subsystem Validation
Isolated manipulator trials yielded an overall grasping success rate of **75%** ($\text{Wilson Score Interval}: 72\% \pm 14\%$). Accuracy varied significantly by target depth:
* **Closest Leaves (Front Canopy):** **90% Success** ($\pm 10\%$)
* **Deepest Leaves (Back Canopy):** **50% Success** ($\pm 22\%$)

---

## Full System Integration & Empirical Validation

```
                                SYSTEM STATE MACHINE
 
+--------------+    Scan Command    +--------------+   Target Found   +--------------+
|              +------------------->+              +----------------->+              |
|  IDLE STATE  |                    | SCANNING PATH|                  | APPROACH LEAF|
|              |<-------------------+              |                  |              |
+--------------+   Scan Complete    +--------------+                  +------+-------+
                                                                             |
                                                                             | In Position
                                                                             v
+--------------+    Retry Count     +--------------+   Pulls Petiole  +--------------+
|              |<-------------------+              |<-----------------+              |
|  RETRACT /   |       < Max        | PRUNING GRASP|                  | EXECUTE PULL |
| EVALUATE MASK|                    |              |                  |              |
+-------+------+                    +--------------+                  +--------------+
        |
        | Retry Exceeded / Success
        v
+--------------+
| RESUME SCAN  |
+--------------+
```

### Integrated Test Setup
The full system integration was evaluated on a physical $1\text{ m}^2$ indoor green wall mockup under ambient indoor environment conditions.
* **Trials Conducted:** 17 full autonomous operations.
* **Total Target Leaves Evaluated:** 85 individual chlorotic leaves.
* **Total Pruning Execution Attempts:** 101 pruning events.

```
+-----------------------------------------------------------------------------------------+
|                                INTEGRATED SYSTEM RESULTS                                |
+-----------------------------------------------------------------------------------------+
| Metric                                              | Value                             |
+-----------------------------------------------------+-----------------------------------+
| Single-Attempt Success Rate (Per Pruning Event)     | **70.3%** (71 / 101)              |
| Multi-Attempt Cumulative Success Rate (Per Leaf)    | **83.5%** (71 / 85)               |
| Median Process Time per Target                      | 14.2 s                            |
+-----------------------------------------------------------------------------------------+
```

### Pareto Analysis of Failure Modes
A statistical breakdown of the 30 failed pruning attempts identified four root causes:

```
+-----------------------------------------------------------------------------------------+
| PARETO FAILURE BREAKDOWN                                                                |
+-----------------------------------------------------------------------------------------+
| Cause                               | Count | Relative % | Cumulative %                 |
+-------------------------------------+-------+------------+------------------------------+
| 1. Pushed Leaf Away                 | 11    | 36.7%      | 36.7%                        |
| 2. Positional Miss                  | 9     | 30.0%      | 66.7%                        |
| 3. Gripper Slippage / Fail          | 6     | 20.0%      | 86.7%                        |
| 4. Other (Occlusions/Deep Depth)    | 4     | 13.3%      | 100.0%                       |
+-----------------------------------------------------------------------------------------+
```

```
 100% +-------------------------------------------------------+--+
      |                                                  *----'  |
  80% |                                            *----'        |  C
      |                                      *----'              |  U
  60% |                                *----'                    |  M
      |                          *----'                          |  U
  40% |                    *----'                                |  L
      |              *----'                                      |  A
  20% |        *----'                                            |  T
      |  *----'                                                  |  I
   0% +--+----------------------+------------------+-------------+--+
         |  Pushed Leaf Away    | Positional Miss  | Gripper Fail|
```

### Statistical Analysis: Key Finding
Hypothesis testing revealed a statistically significant relationship between **image plane target distance ($p$) and pruning success ($p = 0.0429*$)**:
* Targets located **near the optical center** ($\le 167.5\text{ px}$) achieved an **80.4% success rate**.
* Targets located **near the image periphery** ($> 167.5\text{ px}$) dropped to a **60.0% success rate**.

**Root Cause:** This drop in accuracy stems from unmodeled dynamic shifts in the physical cable exit points ($b_i$) on the corner pulleys as the CDPR moves away from its central origin, introducing systematic angular drift into the Forward Kinematic estimator.

---

## Conclusion & Future Work

### Conclusion
This thesis presented the successful design, building, and empirical validation of a proof-of-concept autonomous green wall pruning robot.
1. The **CDPR platform** governed by kinematic control achieved sub-centimeter positioning accuracy across **98.7%** of its workspace.
2. The **Hybrid Perception Pipeline (Mahalanobis + SAM 2)** solved the speed-quality trade-off, operating in **$0.61\text{ s}$** with an **87.7% IoU**.
3. The integrated **Grasp-and-Pull system** proved the viability of fully automated plant maintenance, successfully removing **83.5%** of target yellow leaves.

### Future Work & Scaling Up

```
+-----------------------------------------------------------------------------------+
|                                 FUTURE ROADMAP                                    |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  +--------------------+    +--------------------+    +-------------------------+  |
|  | 1. CDPR Precision  |    | 2. Actuation      |    | 3. Perception & Arm     |  |
|  |                    |    |                    |    |                         |  |
|  | * Dynamic pulley   |    | * Direct-drive BLDC|    | * Visual servoing loop  |  |
|  |   exit-point model |    |   motors (no gear  |    | * Multi-DoF wrist for   |  |
|  | * Onboard IMU pose |    |   friction)        |    |   angled canopy access  |  |
|  |   fusion via EKF   |    | * Active closed-   |    | * Short-range depth     |  |
|  |                    |    |   loop force control|   |   sensor (< 20 cm)      |  |
|  +--------------------+    +--------------------+    +-------------------------+  |
+-----------------------------------------------------------------------------------+
```

1. **Kinematic & Dynamic Model Refinement:** Implement explicit geometric modeling of variable pulley exit vectors ($b_i(\theta)$) to eliminate structural errors near workspace borders. Integrate an IMU on the end effector for Extended Kalman Filtering (EKF) pose estimation.
2. **Direct-Drive Hardware Upgrade:** Transition from high-gear servo motors to low-friction Direct-Drive BLDC motors with active load-cell tension sensing. This will enable real-time force control without friction instability.
3. **In-Hand Visual Servoing:** Introduce a secondary visual servoing control loop on the end effector to compensate for CDPR sway during the final approach phase.
4. **Full-Scale Building Deployment:** Scale the physical frame to the full multi-story ($9\text{ m} \times 15\text{ m}$) green wall layout at SDU, incorporating intermediate cable guide rails.

---

## Installation & Setup Guide

### System Requirements
* **OS:** Ubuntu 22.04 LTS (or Windows 10/11 with WSL2)
* **Python Version:** Python 3.10 or higher
* **CUDA:** Version 11.8 or higher (NVIDIA GPU recommended for SAM 2 inference)
* **Middleware:** ROS 2 Humble / Iron

### Step-by-Step Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/alexellegaard/Greenwall-Pruning.git
   cd Greenwall-Pruning
   ```

2. **Install Dynamixel SDK:**
   > **Important:** Install outside virtual environments if system-wide serial access is required.
   ```bash
   pip install dynamixel_sdk --break-system-packages
   ```

3. **Configure Hardware Serial Permissions:**
   Grant read/write privileges to the Dynamixel USB interface (typically `/dev/ttyUSB0`):
   ```bash
   sudo chmod a+rw /dev/ttyUSB*
   ```
   *(Optional)* To make permissions permanent across reboots, add your user to the `dialout` group:
   ```bash
   sudo usermod -aG dialout $USER
   ```

4. **Install Segment Anything Model 2 (SAM 2):**
   Follow the official installation protocol from Meta Research:
   ```bash
   git clone https://github.com/facebookresearch/sam2.git
   cd sam2
   pip install -e .
   ```

5. **Download Pre-trained Weights & Verify Hardware:**
   Ensure the SAM 2 checkpoint (`sam2_hiera_small.pt` or `sam2_hiera_large.pt`) is placed inside the `checkpoints/` directory within the perception ROS 2 package workspace.

---

## Repository Structure

```
Greenwall-Pruning/
├── config/                  # Kinematic calibration and baseline parameters
├── data/                    # Evaluation benchmark datasets (150 annotated images)
├── docs/                    # Thesis documentation, schematics, and CAD files
├── src/
│   ├── cdpr_controller/     # ROS 2 nodes for Kinematic and Force control
│   ├── leaf_detection/      # Mahalanobis + SAM 2 hybrid vision pipeline
│   └── pruning_gripper/     # End-effector serial interface and sequence logic
├── videos/                  # Demonstration GIFs and local benchmark assets
├── README.md                # Project README file
└── setup.py                 # Workspace build script
```

---

## Authors & Citation

**Authors:**
* **Alex Mejlvang Ellegaard** — [GitHub](https://github.com/alexellegaard) | *MSc Robot Systems, SDU*
* **Anders Lind-Thomsen** — [GitHub](https://github.com/AndersLind-Thomsen) | *MSc Robot Systems, SDU*

**Academic Supervisor:**
* **Oskar Palinko** — *Associate Professor, Maersk Mc-Kinney Moller Institute, SDU*

If you utilize this work or reference our methodology, please cite:

```bibtex
@mastersthesis{EllegaardLindThomsen2026GreenwallPruning,
  author       = {Alex Mejlvang Ellegaard and Anders Lind-Thomsen},
  title        = {Design and Implementation of an Autonomous Robotic System for Pruning Indoor Green Walls},
  school       = {University of Southern Denmark (SDU), Maersk Mc-Kinney Moller Institute},
  year         = {2026},
  month        = {June},
  type         = {Master's Thesis}
}
```