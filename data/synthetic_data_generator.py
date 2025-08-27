import pybullet as p
import pybullet_data
import numpy as np
import random
import os
import yaml
from PIL import Image
import json

class SyntheticDataGenerator:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Connect to physics server
        self.physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Create output directory
        os.makedirs(self.config['data_dir'], exist_ok=True)
        
    def setup_scene(self):
        # Load plane
        p.loadURDF("plane.urdf")
        
        # Load bin
        bin_pos = [0, 0, 0]
        self.bin_id = p.loadURDF("models/bin.urdf", bin_pos)
        
        # Load robot (simplified as static camera for data generation)
        robot_pos = [0.5, 0, 0.5]
        self.robot_id = p.loadURDF("models/ur5.urdf", robot_pos, useFixedBase=True)
        
    def load_objects(self):
        object_ids = []
        objects_in_scene = []
        
        for _ in range(self.config['num_objects']):
            obj_config = random.choice(self.config['objects'])
            obj_variation = {
                'color': random.choice(obj_config['variations']['color']),
                'size': random.choice(obj_config['variations']['size'])
            }
            
            # Random position in bin
            pos = [random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), 0.1]
            orn = p.getQuaternionFromEuler([0, 0, random.uniform(0, 3.14)])
            
            obj_id = p.loadURDF(obj_config['path'], pos, orn, globalScaling=obj_variation['size'])
            object_ids.append(obj_id)
            objects_in_scene.append({
                'name': obj_config['name'],
                'variation': obj_variation,
                'id': obj_id
            })
            
        return object_ids, objects_in_scene
        
    def get_camera_image(self):
        # Get camera parameters from config
        cam_config = self.config['camera']
        
        # Compute view matrix
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_config['eye_position'],
            cameraTargetPosition=cam_config['target_position'],
            cameraUpVector=cam_config['up_vector']
        )
        
        # Compute projection matrix
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=cam_config['fov'],
            aspect=cam_config['aspect'],
            nearVal=cam_config['near_val'],
            farVal=cam_config['far_val']
        )
        
        # Get camera image
        _, _, rgb, _, _ = p.getCameraImage(
            width=cam_config['width'],
            height=cam_config['height'],
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix
        )
        
        # Reshape and convert to PIL Image
        rgb = np.reshape(rgb, (cam_config['height'], cam_config['width'], 4))
        rgb = rgb[:, :, :3]  # Remove alpha channel
        return Image.fromarray(rgb.astype(np.uint8))
    
    def generate_command(self, objects_in_scene):
        template = random.choice(self.config['command_templates'])
        obj = random.choice(objects_in_scene)
        location = random.choice(self.config['locations'])
        
        return template.format(object=obj['name'], location=location['name'])
    
    def plan_trajectory(self, target_obj_id, location):
        # Simplified trajectory planning for data generation
        # In a real scenario, this would use a proper motion planner
        
        # Get object position
        obj_pos, _ = p.getBasePositionAndOrientation(target_obj_id)
        
        # Generate waypoints
        waypoints = [
            [obj_pos[0], obj_pos[1], obj_pos[2] + 0.1],  # Above object
            [obj_pos[0], obj_pos[1], obj_pos[2]],        # At object
            [obj_pos[0], obj_pos[1], obj_pos[2] + 0.1],  # Above object (gripped)
            [location['position'][0], location['position'][1], location['position'][2] + 0.1],  # Above target
            location['position']                          # At target
        ]
        
        # Convert to action sequence
        actions = []
        for wp in waypoints:
            actions.append(f"MOVE_TO {wp[0]} {wp[1]} {wp[2]} 0 0 0")
        
        actions.append("OPEN_GRIPPER")
        actions.append("CLOSE_GRIPPER")
        actions.append("DONE")
        
        return actions
    
    def generate_data(self):
        metadata = []
        
        for episode in range(self.config['num_episodes']):
            # Reset simulation
            p.resetSimulation()
            self.setup_scene()
            
            # Load objects
            object_ids, objects_in_scene = self.load_objects()
            
            # Step simulation to stabilize objects
            for _ in range(100):
                p.stepSimulation()
            
            # Capture image
            image = self.get_camera_image()
            image_path = os.path.join(self.config['data_dir'], f"image_{episode:06d}.png")
            image.save(image_path)
            
            # Generate command
            command = self.generate_command(objects_in_scene)
            
            # Select target object
            target_obj = random.choice(objects_in_scene)
            
            # Plan trajectory (simplified)
            location = random.choice(self.config['locations'])
            actions = self.plan_trajectory(target_obj['id'], location)
            action_str = ' '.join(actions)
            
            # Add to metadata
            metadata.append({
                'image_path': image_path,
                'command': command,
                'actions': action_str,
                'objects': [{'name': obj['name'], 'variation': obj['variation']} for obj in objects_in_scene],
                'target_object': target_obj['name'],
                'target_location': location['name']
            })
            
            if episode % 100 == 0:
                print(f"Generated {episode} episodes")
        
        # Save metadata
        with open(os.path.join(self.config['data_dir'], 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        
        p.disconnect()

if __name__ == "__main__":
    generator = SyntheticDataGenerator("configs/data_config.yaml")
    generator.generate_data()
