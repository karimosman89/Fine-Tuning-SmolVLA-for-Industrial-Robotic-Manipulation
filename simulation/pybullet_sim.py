import pybullet as p
import pybullet_data
import time
import numpy as np
from utils.image_utils import preprocess_image

class PyBulletSim:
    def __init__(self, gui=True):
        if gui:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        
        # Simulation parameters
        self.time_step = 1.0 / 240.0
        self.robot = None
        self.objects = []
        self.camera_config = {
            'eye_position': [0.5, -0.5, 0.5],
            'target_position': [0, 0, 0.1],
            'up_vector': [0, 0, 1],
            'fov': 60,
            'aspect': 1.0,
            'near_val': 0.1,
            'far_val': 100,
            'width': 224,
            'height': 224
        }
    
    def load_environment(self):
        # Load plane
        p.loadURDF("plane.urdf")
        
        # Load bin
        bin_pos = [0, 0, 0]
        self.bin_id = p.loadURDF("models/bin.urdf", bin_pos)
        
        # Load robot (UR5)
        robot_start_pos = [0, 0, 0.5]
        robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF("models/ur5.urdf", robot_start_pos, robot_start_orientation, useFixedBase=True)
        
        # Load objects
        self.load_objects()
        
        # Step simulation to stabilize
        for _ in range(100):
            p.stepSimulation()
    
    def load_objects(self):
        # Load some sample objects
        objects = [
            {"name": "bolt", "path": "models/bolt.urdf", "position": [0, 0, 0.1]},
            {"name": "gear", "path": "models/gear.urdf", "position": [0.1, 0, 0.1]},
            {"name": "bracket", "path": "models/bracket.urdf", "position": [-0.1, 0, 0.1]}
        ]
        
        for obj in objects:
            obj_id = p.loadURDF(obj["path"], obj["position"])
            self.objects.append({
                "id": obj_id,
                "name": obj["name"],
                "position": obj["position"]
            })
    
    def get_camera_image(self):
        # Get camera parameters
        cam_config = self.camera_config
        
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
        return rgb
    
    def execute_action(self, action):
        """
        Execute a single action in the simulation
        """
        if action['type'] == 'MOVE_TO':
            # For simplicity, we'll just set the robot's position directly
            # In a real scenario, you would use inverse kinematics
            target_pos = [action['x'], action['y'], action['z']]
            target_orn = p.getQuaternionFromEuler([action['roll'], action['pitch'], action['yaw']])
            
            # Reset robot base position (simplified)
            p.resetBasePositionAndOrientation(self.robot_id, target_pos, target_orn)
            
        elif action['type'] == 'OPEN_GRIPPER':
            # Open gripper logic
            print("Opening gripper")
            
        elif action['type'] == 'CLOSE_GRIPPER':
            # Close gripper logic
            print("Closing gripper")
        
        # Step simulation
        p.stepSimulation()
        time.sleep(self.time_step)
    
    def execute_action_sequence(self, actions):
        """
        Execute a sequence of actions
        """
        for action in actions:
            self.execute_action(action)
    
    def reset(self):
        """
        Reset simulation
        """
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        self.load_environment()
    
    def close(self):
        """
        Close simulation
        """
        p.disconnect()

if __name__ == "__main__":
    sim = PyBulletSim(gui=True)
    sim.load_environment()
    
    # Example usage
    try:
        while True:
            p.stepSimulation()
            time.sleep(1./240.)
    except KeyboardInterrupt:
        sim.close()
