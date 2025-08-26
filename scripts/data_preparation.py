
import os
import random
import numpy as np
from PIL import Image

# Placeholder for simulation environment libraries (e.g., NVIDIA Isaac Sim, PyBullet, CoppeliaSim)
# In a real scenario, you would integrate with these simulators.

class RoboticSimulation:
    def __init__(self, asset_path="./assets"):
        self.asset_path = asset_path
        self.parts = self._load_assets()
        self.robot_state = {"gripper": "open", "position": [0, 0, 0]}

    def _load_assets(self):
        # In a real simulation, this would load 3D models and their properties
        print(f"Loading assets from {self.asset_path}...")
        return ["hex_bolt", "gear", "silver_M8_bolt", "failed_part"]

    def spawn_parts(self, num_parts=5):
        # Simulate spawning parts in a bin with random properties
        spawned_parts = []
        for _ in range(num_parts):
            part_name = random.choice(self.parts)
            position = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(0, 0.2)]
            material = random.choice(["shiny", "matte"])
            color = [random.randint(0, 255) for _ in range(3)]
            spawned_parts.append({"name": part_name, "position": position, "material": material, "color": color})
        print(f"Spawned {len(spawned_parts)} parts.")
        return spawned_parts

    def get_camera_image(self):
        # Simulate capturing an RGB image from the robot's camera perspective
        # In a real sim, this would render the scene.
        width, height = 256, 256
        image_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(image_array)
        return img

    def execute_action(self, action_sequence):
        # Simulate executing a sequence of robotic actions
        print(f"Executing action sequence: {action_sequence}")
        # This is a simplified execution. In reality, a motion planner would be involved.
        for action in action_sequence.split():
            if action.startswith("MOVE_TO"):
                coords = [float(x) for x in action.split()[1:]]
                self.robot_state["position"] = coords[:3]
                print(f"Robot moved to {self.robot_state['position']}")
            elif action == "OPEN_GRIPPER":
                self.robot_state["gripper"] = "open"
                print("Gripper opened.")
            elif action == "CLOSE_GRIPPER":
                self.robot_state["gripper"] = "closed"
                print("Gripper closed.")
            elif action == "DONE":
                print("Task done.")
                break
        return True # Assume success for simulation


def generate_natural_language_command(parts_in_bin):
    # Generate a natural language command based on parts in the bin
    if not parts_in_bin:
        return "Pick up something."

    target_part = random.choice(parts_in_bin)
    commands = [
        f"Pick up the {target_part['name']}.",
        f"Grab a {target_part['material']} {target_part['name']}.",
        f"Find the {target_part['name']} and place it in the red quality control bin."
    ]
    return random.choice(commands)


def generate_action_sequence(command, parts_in_bin):
    # This is a highly simplified action sequence generation.
    # In a real system, this would come from a motion planner in the simulator.
    if "hex bolt" in command.lower():
        return "MOVE_TO 0.5 0.2 0.3 0 0 1.57 OPEN_GRIPPER MOVE_TO 0.5 0.1 0.25 CLOSE_GRIPPER MOVE_TO 0.6 0.3 0.4 OPEN_GRIPPER DONE"
    elif "gear" in command.lower():
        return "MOVE_TO 0.4 0.1 0.2 0 0 0 OPEN_GRIPPER MOVE_TO 0.4 0.05 0.15 CLOSE_GRIPPER MOVE_TO 0.7 0.2 0.3 OPEN_GRIPPER DONE"
    else:
        return "MOVE_TO 0.5 0.2 0.3 0 0 1.57 OPEN_GRIPPER MOVE_TO 0.5 0.1 0.25 CLOSE_GRIPPER MOVE_TO 0.6 0.3 0.4 OPEN_GRIPPER DONE"


def generate_data(num_examples=1000, output_dir="./data"):
    os.makedirs(output_dir, exist_ok=True)
    sim = RoboticSimulation()
    data_samples = []

    for i in range(num_examples):
        print(f"Generating example {i+1}/{num_examples}...")
        parts_in_bin = sim.spawn_parts(random.randint(3, 7))
        image = sim.get_camera_image()
        language_command = generate_natural_language_command(parts_in_bin)
        action_sequence = generate_action_sequence(language_command, parts_in_bin)

        # Save image
        image_filename = os.path.join(output_dir, f"bin_image_{i:05d}.png")
        image.save(image_filename)

        data_samples.append({
            "image_path": image_filename,
            "language_command": language_command,
            "action_sequence": action_sequence
        })

        # Optional: Execute action in sim to verify (not strictly needed for data generation itself)
        # sim.execute_action(action_sequence)

    # Save metadata (e.g., to a JSON or CSV file)
    import json
    with open(os.path.join(output_dir, "dataset_metadata.json"), "w") as f:
        json.dump(data_samples, f, indent=4)

    print(f"Generated {num_examples} data samples and saved to {output_dir}")


if __name__ == "__main__":
    # Example usage:
    generate_data(num_examples=10, output_dir="./data/synthetic_data")

    # To run a larger generation:
    # generate_data(num_examples=50000, output_dir="./data/synthetic_data")

    print("Data generation script finished.")


