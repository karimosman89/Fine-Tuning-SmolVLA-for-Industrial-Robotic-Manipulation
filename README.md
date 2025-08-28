# SmolVLA-450M for Industrial Robotic Manipulation

This repository contains a professional project plan for fine-tuning the SmolVLA-450M model for industrial robotic manipulation, specifically focusing on adaptive bin picking.

---

## 1. Executive Summary

**Project Title:** Development of a Vision-Language-Control System for Adaptive Bin Picking using SmolVLA-450M.

**Objective:** To fine-tune the SmolVLA-450M model to understand natural language commands and visual scenes to perform precise pick-and-place tasks in a dynamic industrial environment, specifically for bin picking with variable parts.

**Key Value Proposition:** Move beyond pre-programmed robotic motions to an adaptive system that can understand instructions like "Pick up the shiny bolt and place it in the red quality control bin," increasing flexibility and reducing programming time for new tasks.

**Model Chosen:** SmolVLA-450M. It is uniquely suited for this because:
*   **VLA Architecture:** Natively combines visual perception, language understanding, and action generation.
*   **Small Footprint (450M parameters):** Enables deployment on edge devices with limited computational resources (e.g., an NVIDIA Jetson AGX Orin on the robot arm).
*   **Open Weights:** Allows for fine-tuning on proprietary data.

---

## 2. Industrial Use Case Definition: Adaptive Bin Picking

**Problem:** Traditional bin-picking systems are rigid. They require extensive 3D modeling of each new part, meticulous calibration, and complex, hand-coded algorithms for path planning and grasping. They fail with unfamiliar parts, mixed bins, or changing instructions.

**Solution:** A fine-tuned SmolVLA-450M agent that:
1.  **Perceives:** Uses a standard RGB camera to look at a bin of mixed industrial parts (e.g., bolts, gears, brackets).
2.  **Understands:** Interprets a natural language or structured text command from a human operator or a Manufacturing Execution System (MES).
3.  **Acts:** Outputs a sequence of low-level robotic actions (e.g., joint angles, gripper commands) to successfully complete the task.

**Example Commands:**
*   "Pick up the largest gear and place it on the conveyor belt."
*   "Grab a silver M8 bolt and insert it into the fixture on the left."
*   "Remove all the failed parts and drop them in the rejection bin."

---

## 3. Project Phases

### Phase 1: Data Preparation & Synthesis (The Foundation)

This is the most critical phase. We cannot collect millions of real-world robot trials, so we will rely heavily on simulation and data synthesis.

1.  **Simulation Environment:** Use NVIDIA Isaac Sim, PyBullet, or CoppeliaSim to create a digital twin of the robotic workcell (robot arm, gripper, bins, conveyor, parts).
2.  **Asset Creation:** Populate the sim with 3D models of target parts (bolts, gears, etc.). Apply random materials (shiny, matte), colors, and textures to ensure visual diversity.
3.  **Action Representation:** Define the action space. SmolVLA outputs a sequence of tokens. We will define a simple text-based action language:
    *   `MOVE_TO [x] [y] [z] [roll] [pitch] [yaw]`
    *   `OPEN_GRIPPER`
    *   `CLOSE_GRIPPER`
    *   `DONE`
4.  **Automatic Data Generation:**
    *   Randomly spawn parts in the bin.
    *   For each scene, generate a multitude of natural language commands that describe possible tasks (`"pick the [object_name]", "place it in [location_name]"`).
    *   Use a motion planner (e.g., RRT) in simulation to generate the successful trajectory (action sequence) for each (scene, command) pair.
    *   **Key:** Also record the RGB image from the robot\'s camera perspective for each scene.
5.  **Data Format:** Each training example will be a tuple:
    `(image, language_command, action_sequence)`
    Example: `(bin_image_45.png, "Pick up the hex bolt", "MOVE_TO 0.5 0.2 0.3 0 0 1.57 OPEN_GRIPPER MOVE_TO 0.5 0.1 0.25 ... DONE")`
6.  **Volume:** Aim for 50,000 - 100,000+ unique examples.

### Phase 2: Fine-Tuning Setup

1.  **Base Model:** Load the pre-trained `smolvla-450M` weights.
2.  **Framework:** Use a standard LLM fine-tuning framework:
    *   **Primary Choice:** Hugging Face `transformers` + `accelerate` + `peft` (for Parameter-Efficient Fine-Tuning).
    *   **Alternative:** NVIDIA NeMo.
3.  **Fine-Tuning Strategy:**
    *   **Full Fine-Tuning (if compute-rich):** Update all 450M parameters. This is most effective but computationally expensive.
    *   **Parameter-Efficient Fine-Tuning - LoRA (Recommended for initial testing):** Use Low-Rank Adaptation (LoRA) to only train a small set of additional parameters. This is faster, cheaper, and reduces the risk of catastrophic forgetting. We can later switch to full fine-tuning for final performance.
4.  **Infrastructure:** Use a cloud GPU instance (e.g., AWS EC2 `g5.12xlarge` with 4x A10G, or a `p4d.24xlarge` for full fine-tuning).

### Phase 3: Training & Evaluation

1.  **Training Loop:**
    *   **Input:** Tokenized language command + image features (processed by the model\'s vision encoder).
    *   **Target:** Tokenized action sequence.
    *   **Loss Function:** Standard cross-entropy loss on the action tokens (autoregressive prediction).
2.  **Hyperparameters:**
    *   Batch Size: As large as VRAM allows (32-128).
    *   Learning Rate: Low (~1e-5 to 5e-5 for full FT, ~1e-4 for LoRA).
    *   Schedule: Linear Warmup + Cosine Decay.
3.  **Validation:**
    *   Hold out 10% of the synthetic data for validation.
    *   **Primary Metric: Action Success Rate (ASR):** The percentage of predicted action sequences that, when executed in the simulator, successfully complete the task.
    *   **Secondary Metric:** Token-level accuracy (BLEU score for actions), but this is less important than actual functional success.

### Phase 4: Simulation Testing & Deployment

1.  **Benchmarking:** Create a challenging test set in simulation with novel object arrangements and commands not seen during training. Measure the ASR.
2.  **Sim-to-Real Transfer:**
    *   **Domain Randomization:** During data generation, randomize lighting, camera noise, and background textures to bridge the sim-to-real gap.
    *   **Deploy the model to a real robot** (e.g., UR5, Franka Emika) with an NVIDIA Jetson module.
    *   **Create a safety wrapper:** The robot\'s low-level controller must monitor the predicted actions for feasibility and safety (e.g., check for collisions, joint limits) before execution.
3.  **Deployment Architecture:**
    *   The fine-tuned model runs on the Jetson.
    *   It takes an image from an overhead camera and a text command from the network.
    *   It streams predicted action tokens.
    *   A ROS node (Robot Operating System) parses these tokens into joint goal positions and publishes them to the robot controller.

### Phase 5: Iteration & Improvement

1.  **Identify Failure Modes:** Analyze where the model fails in the real world.
2.  **Data Augmentation:** Collect a small amount of real-world data (e.g., 100 examples of failed tasks, now corrected) and add them to the training set. This is often enough to correct specific problems.
3.  **Active Learning:** Have the robot flag scenarios where it is least confident in its predictions for human intervention and re-training.

---

## 4. Technical Considerations & Challenges

*   **Sim-to-Real Gap:** This is the biggest challenge. Extensive domain randomization is non-negotiable.
*   **Action Representation:** The choice of action space is crucial. A continuous `(x, y, z, gripper)` space might be easier to learn than joint angles. The text-based action tokens must be parsed reliably.
*   **Safety:** **A robot is a powerful and dangerous tool.** The model\'s outputs **must never** be sent directly to the robot. A rigorous, rule-based safety supervisor is required to validate all actions.
*   **Compute Costs:** Fine-tuning, even a small model, requires significant GPU resources. Budget accordingly.

---

## 5. Conclusion

This project outlines a feasible and high-impact application of small-scale foundation models in industrial robotics. By leveraging simulation for data generation and parameter-efficient fine-tuning techniques, we can adapt the general-purpose SmolVLA-450M into a specialized, safe, and effective control policy for complex tasks like adaptive bin picking, paving the way for more flexible and intelligent factories.





## 6. Project Structure

This repository is structured as follows:

```
.
├── data/
│   └── synthetic_data/  # Contains generated images and dataset_metadata.json
├── models/
│   └── fine_tuned_smolvla/ # Stores the fine-tuned SmolVLA model weights
├── scripts/
│   ├── data_preparation.py  # Script for generating synthetic data
│   ├── fine_tune.py         # Script for fine-tuning the SmolVLA model
│   └── evaluate.py          # Script for evaluating the fine-tuned model
├── config/                  # Placeholder for configuration files (e.g., model hyperparameters)
├── .gitignore               # Specifies intentionally untracked files to ignore
├── README.md                # Project overview and instructions
└── requirements.txt         # Python dependencies
```

## 7. Getting Started

Follow these steps to set up the environment, generate synthetic data, fine-tune the model, and evaluate its performance.

### 7.1. Environment Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/karimosman89/Fine-Tuning-SmolVLA-450M-for-Industrial-Robotic-Manipulation.git
    cd Fine-Tuning-SmolVLA-450M-for-Industrial-Robotic-Manipulation
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 7.2. Data Preparation

Generate synthetic data for training and evaluation. This script simulates a robotic environment and creates image-language-action sequence tuples.

```bash
python scripts/data_preparation.py
```

By default, this will generate 10 examples in `./data/synthetic_data`. You can modify the `num_examples` variable in `scripts/data_preparation.py` to generate a larger dataset (e.g., 50,000 to 100,000+ examples as suggested in the project plan).

### 7.3. Model Fine-Tuning

Fine-tune the SmolVLA-450M model using the generated synthetic data. The script supports Parameter-Efficient Fine-Tuning (LoRA) for faster and more efficient training.

```bash
python scripts/fine_tune.py
```

The fine-tuned model will be saved in the `./models/fine_tuned_smolvla` directory.

### 7.4. Model Evaluation

Evaluate the performance of the fine-tuned model using the evaluation script. This script calculates the Action Success Rate (ASR) by simulating the execution of predicted action sequences.

```bash
python scripts/evaluate.py
```

## 8. Future Work and Real-World Deployment

This project provides a strong foundation for developing a robust vision-language-control system for industrial robotics. Future work will involve:

*   **Advanced Simulation:** Integrating with more sophisticated robotic simulators (e.g., NVIDIA Isaac Sim) for more realistic data generation and sim-to-real transfer studies.
*   **Real-World Data Collection:** Gradually incorporating real-world robot interaction data to bridge the sim-to-real gap and improve robustness.
*   **Safety Protocols:** Implementing rigorous safety wrappers and monitoring systems to ensure safe operation of the robot based on the model's outputs.
*   **Deployment Optimization:** Optimizing the model for deployment on edge devices like NVIDIA Jetson AGX Orin for real-time inference.

---

**Author:** Karim Osman


