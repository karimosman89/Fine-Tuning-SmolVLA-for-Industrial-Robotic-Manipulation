import re
import numpy as np

def parse_action_string(action_string):
    """
    Parse action string into structured format
    """
    actions = []
    lines = action_string.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('MOVE_TO'):
            # Parse coordinates
            coords = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            if len(coords) >= 3:
                action = {
                    'type': 'MOVE_TO',
                    'x': float(coords[0]),
                    'y': float(coords[1]),
                    'z': float(coords[2]),
                    'roll': float(coords[3]) if len(coords) > 3 else 0.0,
                    'pitch': float(coords[4]) if len(coords) > 4 else 0.0,
                    'yaw': float(coords[5]) if len(coords) > 5 else 0.0
                }
                actions.append(action)
        elif line == 'OPEN_GRIPPER':
            actions.append({'type': 'OPEN_GRIPPER'})
        elif line == 'CLOSE_GRIPPER':
            actions.append({'type': 'CLOSE_GRIPPER'})
        elif line == 'DONE':
            actions.append({'type': 'DONE'})
            
    return actions

def action_dict_to_string(action_dict):
    """
    Convert action dictionary to string
    """
    if action_dict['type'] == 'MOVE_TO':
        return (f"MOVE_TO {action_dict['x']} {action_dict['y']} {action_dict['z']} "
                f"{action_dict['roll']} {action_dict['pitch']} {action_dict['yaw']}")
    else:
        return action_dict['type']

def interpolate_waypoints(start, end, num_points=10):
    """
    Interpolate between waypoints
    """
    waypoints = []
    for i in range(num_points):
        alpha = i / (num_points - 1)
        waypoint = {
            'x': start['x'] + alpha * (end['x'] - start['x']),
            'y': start['y'] + alpha * (end['y'] - start['y']),
            'z': start['z'] + alpha * (end['z'] - start['z']),
            'roll': start['roll'] + alpha * (end['roll'] - start['roll']),
            'pitch': start['pitch'] + alpha * (end['pitch'] - start['pitch']),
            'yaw': start['yaw'] + alpha * (end['yaw'] - start['yaw'])
        }
        waypoints.append(waypoint)
    
    return waypoints

def check_collision(waypoint, obstacles):
    """
    Check if a waypoint collides with obstacles
    """
    for obstacle in obstacles:
        # Simple sphere collision check
        distance = np.sqrt(
            (waypoint['x'] - obstacle['x'])**2 +
            (waypoint['y'] - obstacle['y'])**2 +
            (waypoint['z'] - obstacle['z'])**2
        )
        
        if distance < obstacle['radius']:
            return True
    
    return False
