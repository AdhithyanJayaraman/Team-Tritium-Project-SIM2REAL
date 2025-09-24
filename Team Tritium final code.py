import numpy as np
import socket
import threading
import time
import json
from PyFlyt.core import Aviary
import queue
import random
import pybullet as p
import math

class MovingTarget:
    def __init__(self, initial_position, velocity, bounds):
        self.position = np.array(initial_position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.bounds = bounds  # [x_min, x_max, y_min, y_max, z_min, z_max]
        self.marker_id = None

    def update_position(self, dt):
        self.position += self.velocity * dt
        
        # Check X boundary (4m width: 0 to 4)
        if self.position[0] <= self.bounds[0] or self.position[0] >= self.bounds[1]:
            self.velocity[0] *= -1
            self.position[0] = np.clip(self.position[0], self.bounds[0], self.bounds[1])
        
        # Check Y boundary (5m height: 0 to 5)
        if self.position[1] <= self.bounds[2] or self.position[1] >= self.bounds[3]:
            self.velocity[1] *= -1
            self.position[1] = np.clip(self.position[1], self.bounds[2], self.bounds[3])
        
        # Check Z boundary (floor level)
        if self.position[2] <= self.bounds[4] or self.position[2] >= self.bounds[5]:
            self.velocity[2] *= -1
            self.position[2] = np.clip(self.position[2], self.bounds[4], self.bounds[5])

    def get_position(self):
        return self.position.copy()

class DroneSwarmCommunication:
    def __init__(self, drone_id, num_drones=5, base_port=12000, communication_range=14.0):
        self.drone_id = drone_id
        self.num_drones = num_drones
        self.base_port = base_port
        self.communication_range = communication_range
        self.alpha = 0.01

        self.my_port = base_port + drone_id
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('localhost', self.my_port))
        self.socket.settimeout(0.1)

        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.obstacles = []

        self.neighbor_states = {}
        self.running = True
        self.comm_thread = threading.Thread(target=self._communication_loop, daemon=True)
        self.comm_thread.start()

        # Swarm-target shared variables
        self.target_found = False
        self.target_position = None

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles

    def _communication_loop(self):
        while self.running:
            try:
                data, addr = self.socket.recvfrom(1024)
                message = json.loads(data.decode())
                sender_id = message['drone_id']
                if sender_id != self.drone_id:
                    self.neighbor_states[sender_id] = {
                        'position': np.array(message['position']),
                        'velocity': np.array(message['velocity']),
                        'timestamp': time.time()
                    }
                    # Check if detection signal received
                    if message.get('target_found', False):
                        self.target_found = True
                        self.target_position = np.array(message['target_position'])
            except socket.timeout:
                continue

    def broadcast_state(self, target_found=False, target_position=None):
        message = {
            'drone_id': self.drone_id,
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'timestamp': time.time(),
            'target_found': target_found,
            'target_position': target_position.tolist() if target_position is not None else None
        }
        message_data = json.dumps(message).encode()
        for other_drone_id in range(self.num_drones):
            if other_drone_id != self.drone_id:
                try:
                    target_port = self.base_port + other_drone_id
                    self.socket.sendto(message_data, ('localhost', target_port))
                except Exception:
                    pass

    def update_state(self, position, velocity):
        self.position = np.array(position)
        self.velocity = np.array(velocity)

    def close(self):
        self.running = False
        self.socket.close()
        if self.comm_thread.is_alive():
            self.comm_thread.join(timeout=1.0)

class DroneSwarmSimulation:
    def __init__(self, num_drones=5):
        self.num_drones = num_drones
        self.dt = 1.0/240.0
        # Different deployment points across the 4x5 boundary
        self.start_pos = np.array([
            [0.8, 1.2, 1.0],   # Bottom left
            [3.2, 1.2, 1.0],   # Bottom right
            [2.0, 4.2, 1.0],   # Top center
            [0.8, 3.8, 1.0],   # Top left
            [3.2, 3.8, 1.0]    # Top right
        ])
        self.start_orn = np.zeros((num_drones, 3))
        self.obstacles = self._generate_random_obstacles()
        self.env = Aviary(
            start_pos=self.start_pos,
            start_orn=self.start_orn,
            render=True,
            drone_type="quadx"
        )
        self.env.set_mode(6)
        self.obstacle_ids = self._add_visible_obstacles()
        # Single moving target for all drones
        self.moving_target = self._initialize_moving_target()
        self.target_marker_id = self._add_target_marker()
        self.drone_comms = []
        for i in range(num_drones):
            comm = DroneSwarmCommunication(drone_id=i, num_drones=num_drones)
            comm.set_obstacles(self.obstacles)
            self.drone_comms.append(comm)
        self.target_threshold = 1.5  # All drones must be within this distance
        
        # Anti-stuck mechanism
        self.drone_stuck_counters = [0] * num_drones
        self.drone_last_positions = [np.array([0., 0., 0.])] * num_drones
        self.stuck_threshold = 25  # frames to consider stuck (reduced)

    def _initialize_moving_target(self):
        # 4x5 rectangular boundary: x=[0,4], y=[0,5], z=[0,2]
        bounds = [0.0, 4.0, 0.0, 5.0, 0.0, 0.1]  # Target on floor level
        initial_position = [2.0, 0.5, 0.05]  # Start in center on floor within boundary
        velocity = [0.8, 0.5, 0.0]  # Slightly slower target for better tracking
        return MovingTarget(initial_position, velocity, bounds)

    def _generate_random_obstacles(self):
        obstacles = []
        # Smaller obstacles within 4x5 boundary at different locations
        obstacle_configs = [
            ([1.2, 1.8, 0.2], 0.25),  # Reduced size
            ([2.8, 3.2, 0.2], 0.20),  # Reduced size
            ([3.2, 1.5, 0.2], 0.22),  # Reduced size
            ([0.8, 4.2, 0.2], 0.25),  # Reduced size
            ([3.5, 3.8, 0.2], 0.18),  # Reduced size
            ([1.5, 3.0, 0.2], 0.23),  # Additional obstacle
        ]
        for pos, radius in obstacle_configs:
            obstacles.append((np.array(pos), radius))
        return obstacles

    def _add_visible_obstacles(self):
        obstacle_ids = []
        physics_client = getattr(self.env, '_p', getattr(self.env, 'BC', p))
        for i, (obstacle_pos, obstacle_radius) in enumerate(self.obstacles):
            try:
                visual_shape_id = physics_client.createVisualShape(
                    shapeType=p.GEOM_SPHERE,
                    radius=obstacle_radius,
                    rgbaColor=[0.8, 0.1, 0.1, 0.9]
                )
                # Remove collision shape to avoid PyFlyt contact_array issues
                obstacle_id = physics_client.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=visual_shape_id,
                    basePosition=obstacle_pos.tolist()
                )
                obstacle_ids.append(obstacle_id)
            except Exception:
                pass
        return obstacle_ids

    def _add_target_marker(self):
        physics_client = getattr(self.env, '_p', getattr(self.env, 'BC', p))
        try:
            visual_shape_id = physics_client.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=0.23,
                rgbaColor=[0.1, 0.8, 0.1, 0.8]
            )
            marker_id = physics_client.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=self.moving_target.get_position().tolist()
            )
            self.moving_target.marker_id = marker_id
            return marker_id
        except Exception:
            return None

    def _update_target_marker(self):
        physics_client = getattr(self.env, '_p', getattr(self.env, 'BC', p))
        if self.moving_target.marker_id is not None:
            try:
                physics_client.resetBasePositionAndOrientation(
                    self.moving_target.marker_id,
                    self.moving_target.get_position().tolist(),
                    [0, 0, 0, 1]
                )
            except Exception:
                pass

    def _check_and_handle_stuck_drones(self, drone_positions, step):
        """Detect and unstuck drones that have stopped moving"""
        for i in range(self.num_drones):
            # Check if drone has moved significantly
            position_change = np.linalg.norm(drone_positions[i] - self.drone_last_positions[i])
            
            if position_change < 0.03:  # Even smaller movement threshold
                self.drone_stuck_counters[i] += 1
            else:
                self.drone_stuck_counters[i] = 0
            
            # If stuck for too long, apply unstuck force
            if self.drone_stuck_counters[i] > self.stuck_threshold:
                print(f"Step {step}: Drone {i} appears stuck, applying unstuck maneuver")
                # Force movement towards target with some randomness
                target_pos = self.moving_target.get_position()
                unstuck_direction = target_pos - drone_positions[i]
                unstuck_distance = np.linalg.norm(unstuck_direction)
                
                if unstuck_distance > 0.1:
                    # Add random component to avoid repeated stuck situations
                    random_angle = random.uniform(0, 2*np.pi)
                    random_component = 0.3 * np.array([np.cos(random_angle), np.sin(random_angle), 0.0])
                    unstuck_velocity = (unstuck_direction / unstuck_distance) * 1.5 + random_component
                else:
                    # Random movement if too close to target
                    angle = random.uniform(0, 2*np.pi)
                    unstuck_velocity = np.array([np.cos(angle), np.sin(angle), 0.0]) * 1.2
                
                self.drone_comms[i].velocity = unstuck_velocity
                self.drone_stuck_counters[i] = 0  # Reset counter
            
            # Update last position
            self.drone_last_positions[i] = drone_positions[i].copy()

    def _apply_boundary_constraints(self, velocity, position):
        """Apply soft boundary constraints without stopping the drone"""
        bounds = [0.3, 3.7, 0.3, 4.7, 0.3, 1.7]  # Safe zone
        constrained_velocity = velocity.copy()
        
        # Soft boundary repulsion
        repulsion_strength = 1.5
        
        # X boundaries
        if position[0] < bounds[0]:
            constrained_velocity[0] += repulsion_strength * (bounds[0] - position[0])
        elif position[0] > bounds[1]:
            constrained_velocity[0] -= repulsion_strength * (position[0] - bounds[1])
        
        # Y boundaries  
        if position[1] < bounds[2]:
            constrained_velocity[1] += repulsion_strength * (bounds[2] - position[1])
        elif position[1] > bounds[3]:
            constrained_velocity[1] -= repulsion_strength * (position[1] - bounds[3])
        
        # Z boundaries
        if position[2] < bounds[4]:
            constrained_velocity[2] += repulsion_strength * (bounds[4] - position[2])
        elif position[2] > bounds[5]:
            constrained_velocity[2] -= repulsion_strength * (position[2] - bounds[5])
        
        return constrained_velocity

    def _print_target_distances(self, drone_positions, step):
        distances = []
        tpos = self.moving_target.get_position()
        for i in range(self.num_drones):
            distances.append(np.linalg.norm(drone_positions[i] - tpos))
        avg_distance = np.mean(distances)
        print(f"Step {step}: Avg distance = {avg_distance:.2f} | Drones: " +
              ", ".join(f"{dist:.2f}" for dist in distances))
        return distances

    def run_simulation(self, max_steps=12000):
        target_found = False
        
        for step in range(max_steps):
            self.moving_target.update_position(self.dt)
            self._update_target_marker()
            drone_positions = []
            
            # Get current drone positions
            for i in range(self.num_drones):
                state = self.env.state(i)
                pos = np.array(state[3])
                drone_positions.append(pos)
                self.drone_comms[i].update_state(pos, self.drone_comms[i].velocity)
            
            # Check for stuck drones and handle them
            self._check_and_handle_stuck_drones(drone_positions, step)
            
            # Check if target is detected
            if not target_found:
                for i in range(self.num_drones):
                    drone = self.drone_comms[i]
                    distance = np.linalg.norm(drone_positions[i] - self.moving_target.get_position())
                    if distance < 1.8:  # Slightly larger detection range for slower speeds
                        drone.target_found = True
                        drone.target_position = self.moving_target.get_position()
                        drone.broadcast_state(target_found=True, target_position=self.moving_target.get_position())
                        target_found = True
                        print(f"Drone {i} has detected the target and signaled swarm!")
                        break
                
                # Search movement - reduced velocities
                for i in range(self.num_drones):
                    drone = self.drone_comms[i]
                    target_pos = self.moving_target.get_position()
                    
                    # Direct movement towards target with reduced speed
                    direction_to_target = target_pos - drone_positions[i]
                    distance_to_target = np.linalg.norm(direction_to_target)
                    
                    if distance_to_target > 0.1:
                        # Reduced search velocity for better target recognition
                        desired_velocity = (direction_to_target / distance_to_target) * min(1.8, distance_to_target * 1.0)
                    else:
                        # Small movement if very close
                        angle = i * (2*np.pi/self.num_drones) + step * 0.01  # Slight rotation
                        desired_velocity = np.array([np.cos(angle), np.sin(angle), 0.0]) * 0.5
                    
                    # Apply boundary constraints
                    constrained_velocity = self._apply_boundary_constraints(desired_velocity, drone_positions[i])
                    
                    # Smooth velocity update with guaranteed minimum movement
                    drone.velocity = drone.velocity * 0.4 + constrained_velocity * 0.6
                    
                    # Ensure minimum speed to prevent stopping - reduced minimum speed
                    vel_mag = np.linalg.norm(drone.velocity)
                    min_speed = 0.4  # Reduced minimum speed
                    if vel_mag < min_speed:
                        if vel_mag > 0.05:
                            drone.velocity = (drone.velocity / vel_mag) * min_speed
                        else:
                            # Fallback: move towards target at minimum speed
                            if distance_to_target > 0:
                                drone.velocity = (direction_to_target / distance_to_target) * min_speed
                            else:
                                # Circular motion if at target
                                angle = i * (2*np.pi/self.num_drones) + step * 0.02
                                drone.velocity = np.array([np.cos(angle), np.sin(angle), 0.0]) * min_speed
                    
                    # Apply reduced speed limit for search
                    vel_mag = np.linalg.norm(drone.velocity)
                    max_search_speed = 2.0  # Reduced maximum search speed
                    if vel_mag > max_search_speed:
                        drone.velocity = (drone.velocity / vel_mag) * max_search_speed
                    
                    vel_command = np.array([drone.velocity[0], drone.velocity[1], drone.velocity[2], 0.0])
                    self.env.set_setpoint(i, vel_command)
            else:
                # Target following mode - reduced velocities for better tracking
                tpos = self.moving_target.get_position()
                tvel = self.moving_target.velocity
                
                # Reduced prediction time for more responsive tracking
                prediction_time = 0.5
                predicted_target_pos = tpos + tvel * prediction_time
                
                angle_interval = 2*np.pi/self.num_drones
                surround_radius = 0.8
                
                for i in range(self.num_drones):
                    drone = self.drone_comms[i]
                    
                    # Calculate desired formation position
                    theta = i * angle_interval
                    formation_offset = surround_radius * np.array([np.cos(theta), np.sin(theta), 0.0])
                    desired_pos = predicted_target_pos + formation_offset
                    
                    # Direction to desired position
                    direction = desired_pos - drone_positions[i]
                    distance = np.linalg.norm(direction)
                    
                    # Calculate pursuit velocity - reduced speeds
                    if distance > 0.1:
                        pursuit_velocity = (direction / distance) * min(2.5, distance * 1.8)  # Reduced multiplier
                    else:
                        pursuit_velocity = np.zeros(3)
                    
                    # Add target velocity for coordinated movement - reduced coupling
                    target_velocity_compensation = tvel * 1.0  # Reduced from 1.3
                    
                    # Combine velocities
                    desired_velocity = pursuit_velocity + target_velocity_compensation
                    
                    # Apply boundary constraints
                    constrained_velocity = self._apply_boundary_constraints(desired_velocity, drone_positions[i])
                    
                    # Smooth velocity update with momentum
                    momentum = drone.velocity * 0.3
                    drone.velocity = momentum + constrained_velocity * 0.7
                    
                    # Ensure minimum movement speed - reduced minimum
                    vel_mag = np.linalg.norm(drone.velocity)
                    min_follow_speed = 0.6  # Reduced minimum follow speed
                    if vel_mag < min_follow_speed:
                        if vel_mag > 0.1:
                            drone.velocity = (drone.velocity / vel_mag) * min_follow_speed
                        else:
                            # Emergency: move directly towards target
                            target_direction = tpos - drone_positions[i]
                            target_dist = np.linalg.norm(target_direction)
                            if target_dist > 0:
                                drone.velocity = (target_direction / target_dist) * min_follow_speed
                    
                    # Apply reduced maximum speed limit
                    vel_mag = np.linalg.norm(drone.velocity)
                    max_follow_speed = 3.0  # Reduced from 4.5
                    if vel_mag > max_follow_speed:
                        drone.velocity = (drone.velocity / vel_mag) * max_follow_speed
                    
                    vel_command = np.array([drone.velocity[0], drone.velocity[1], drone.velocity[2], 0.0])
                    self.env.set_setpoint(i, vel_command)
            
            self.env.step()
            distances = self._print_target_distances(drone_positions, step)
            
            # Check if ALL drones are within threshold distance
            if target_found and all(d < self.target_threshold for d in distances):
                print(f"SUCCESS! All drones are within {self.target_threshold}m of target at step {step}!")
                print("Mission accomplished! Adding visualization steps...")
                
                # Add visualization steps with formation maintenance
                for viz_step in range(200):  # 200 steps for better visualization
                    self.moving_target.update_position(self.dt)
                    self._update_target_marker()
                    
                    # Get updated positions
                    for i in range(self.num_drones):
                        state = self.env.state(i)
                        pos = np.array(state[3])
                        drone_positions[i] = pos
                    
                    # Maintain formation around target during visualization
                    tpos = self.moving_target.get_position()
                    tvel = self.moving_target.velocity
                    
                    for i in range(self.num_drones):
                        drone = self.drone_comms[i]
                        
                        # Maintain circular formation
                        theta = i * (2*np.pi/self.num_drones)
                        formation_pos = tpos + 0.8 * np.array([np.cos(theta), np.sin(theta), 0.0])
                        
                        # Gentle movement to maintain formation
                        direction = formation_pos - drone_positions[i]
                        distance = np.linalg.norm(direction)
                        
                        if distance > 0.1:
                            maintain_velocity = (direction / distance) * min(1.5, distance * 2.0)
                        else:
                            maintain_velocity = np.zeros(3)
                        
                        # Add target velocity
                        maintain_velocity += tvel * 0.8
                        
                        # Smooth update
                        drone.velocity = drone.velocity * 0.5 + maintain_velocity * 0.5
                        
                        # Speed limits
                        vel_mag = np.linalg.norm(drone.velocity)
                        if vel_mag > 2.0:
                            drone.velocity = (drone.velocity / vel_mag) * 2.0
                        
                        vel_command = np.array([drone.velocity[0], drone.velocity[1], drone.velocity[2], 0.0])
                        self.env.set_setpoint(i, vel_command)
                    
                    self.env.step()
                    
                    if viz_step % 20 == 0:  # Print every 20 steps during visualization
                        self._print_target_distances(drone_positions, step + viz_step)
                
                print("Simulation completed with extended visualization!")

                break
        
        self.cleanup()

    def cleanup(self):
        for comm in self.drone_comms:
            comm.close()
        physics_client = getattr(self.env, '_p', getattr(self.env, 'BC', p))
        for obstacle_id in getattr(self, "obstacle_ids", []):
            try:
                physics_client.removeBody(obstacle_id)
            except Exception:
                pass
        if getattr(self.moving_target, "marker_id", None):
            try:
                physics_client.removeBody(self.moving_target.marker_id)
            except Exception:
                pass
        try:
            self.env.close()
        except Exception:
            pass

def main():
    print("=== DRONE SWARM (Single-Target Search and Surround) ===")
    simulation = DroneSwarmSimulation(num_drones=5)
    try:
        simulation.run_simulation(max_steps=12000)
    except KeyboardInterrupt:
        simulation.cleanup()

if __name__ == "__main__":
    main()