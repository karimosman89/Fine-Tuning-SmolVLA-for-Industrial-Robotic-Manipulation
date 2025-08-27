import re

class SafetyWrapper:
    def __init__(self):
        # Define safety constraints
        self.workspace_limits = {
            'x': [-1.0, 1.0],
            'y': [-1.0, 1.0],
            'z': [0.0, 1.5]
        }
        
        self.max_velocity = 0.5  # m/s
        self.max_acceleration = 2.0  # m/s²
        
    def validate_position(self, x, y, z):
        """Check if position is within workspace limits"""
        return (self.workspace_limits['x'][0] <= x <= self.workspace_limits['x'][1] and
                self.workspace_limits['y'][0] <= y <= self.workspace_limits['y'][1] and
                self.workspace_limits['z'][0] <= z <= self.workspace_limits['z'][1])
    
    def validate_trajectory(self, waypoints):
        """Check if trajectory is safe"""
        if not waypoints:
            return False
        
        # Check each waypoint
        for wp in waypoints:
            if not self.validate_position(wp[0], wp[1], wp[2]):
                return False
        
        # Check velocity and acceleration (simplified)
        # In a real implementation, this would be more thorough
        return True
    
    def parse_actions(self, action_text):
        """Parse action text into structured format"""
        actions = []
        lines = action_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('MOVE_TO'):
                # Parse coordinates
                coords = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                if len(coords) >= 3:
                    actions.append({
                        'type': 'MOVE_TO',
                        'x': float(coords[0]),
                        'y': float(coords[1]),
                        'z': float(coords[2]),
                        'roll': float(coords[3]) if len(coords) > 3 else 0,
                        'pitch': float(coords[4]) if len(coords) > 4 else 0,
                        'yaw': float(coords[5]) if len(coords) > 5 else 0
                    })
            elif line == 'OPEN_GRIPPER':
                actions.append({'type': 'OPEN_GRIPPER'})
            elif line == 'CLOSE_GRIPPER':
                actions.append({'type': 'CLOSE_GRIPPER'})
            elif line == 'DONE':
                actions.append({'type': 'DONE'})
                
        return actions
    
    def validate(self, action_text):
        """Validate and potentially modify actions for safety"""
        actions = self.parse_actions(action_text)
        safe_actions = []
        
        waypoints = []
        for action in actions:
            if action['type'] == 'MOVE_TO':
                waypoints.append([action['x'], action['y'], action['z']])
            
            safe_actions.append(action)
        
        # Validate trajectory
        if not self.validate_trajectory(waypoints):
            # Generate a safe alternative trajectory
            safe_waypoints = self.generate_safe_trajectory(waypoints)
            
            # Replace movement actions with safe ones
            safe_actions = []
            for i, wp in enumerate(safe_waypoints):
                safe_actions.append({
                    'type': 'MOVE_TO',
                    'x': wp[0],
                    'y': wp[1],
                    'z': wp[2],
                    'roll': 0,
                    'pitch': 0,
                    'yaw': 0
                })
            
            # Add gripper actions if needed
            if any(a['type'] in ['OPEN_GRIPPER', 'CLOSE_GRIPPER'] for a in actions):
                for action in actions:
                    if action['type'] in ['OPEN_GRIPPER', 'CLOSE_GRIPPER']:
                        safe_actions.append(action)
            
            safe_actions.append({'type': 'DONE'})
        
        # Convert back to text format
        action_lines = []
        for action in safe_actions:
            if action['type'] == 'MOVE_TO':
                action_lines.append(
                    f"MOVE_TO {action['x']} {action['y']} {action['z']} " +
                    f"{action['roll']} {action['pitch']} {action['yaw']}"
                )
            else:
                action_lines.append(action['type'])
        
        return '\n'.join(action_lines)
    
    def generate_safe_trajectory(self, waypoints):
        """Generate a safe trajectory through waypoints"""
        safe_waypoints = []
        
        for wp in waypoints:
            # Constrain each waypoint to safe workspace
            safe_wp = [
                max(self.workspace_limits['x'][0], min(self.workspace_limits['x'][1], wp[0])),
                max(self.workspace_limits['y'][0], min(self.workspace_limits['y'][1], wp[1])),
                max(self.workspace_limits['z'][0], min(self.workspace_limits['z'][1], wp[2]))
            ]
            safe_waypoints.append(safe_wp)
        
        return safe_waypoints
