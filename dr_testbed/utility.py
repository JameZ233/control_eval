# from dr_testbed.simulator import Simulator
from dr_testbed.render import Renderer
import numpy as np
import pygame
import json
import random

class WayPoint:
    """
    A class to represent a waypoint.
    """
    def __init__(self, 
                 pos,
                 dir, 
                 ):
        """
        Args:
            pos (tuple): position of the waypoint.
            dir (tuple): direction of the waypoint.
        """
        assert is_coordinate(pos), "Position is not a valid position vector"
        assert is_coordinate(dir), "Direction is not a valid direction vector"

        self.pos = pos
        self.dir = (dir[0], dir[1])
        
        self.next_waypoints = []

    def set_next_waypoints(self, next_waypoints):
        for waypoint in next_waypoints:
            assert isinstance(waypoint, WayPoint), "Next waypoint is not a \
                Waypoint object"

        self.next_waypoints = next_waypoints

def is_coordinate(coordinate):
    """
    Check if a coordinate is valid.
    
    Args:
        coordinate (tuple): the coordinate to check.
        
    Returns:
        bool: True if the coordinate is valid, False otherwise.
    """
    if not isinstance(coordinate, tuple):
        return False
    if len(coordinate) != 2:
        return False
    if not all(isinstance(i, float) for i in coordinate):
        return False
    return True

def load_waypoints(filename):
    with open(filename) as f:
        waypoint_map = json.load(f)
    #loaded file is now a nested dictionary
    waypoints = []
    for i in range(1, len(waypoint_map)+1):
        pos = tuple(waypoint_map[str(i)]["pos"])
        ori = tuple(waypoint_map[str(i)]["ori"])
        waypoint = WayPoint(pos, ori)
        waypoints.append(waypoint)
    for i in range(1, len(waypoints)+1):
        next_waypoint_locs = waypoint_map[str(i)]["connections"]
        next_waypoints = []
        #print(next_waypoint_locs)
        for loc in next_waypoint_locs:
            next_waypoints.append(waypoints[loc-1])
        waypoints[i-1].set_next_waypoints(next_waypoints)
    return waypoints

def calc_traj(point_0, point_1, result_type = 'list'):
    """
    Calculate the trajectory between two waypoints.
    
    Args:
        point_0 (WayPoint): the first waypoint.
        point_1 (WayPoint): the second waypoint.
        result_type (str): the type of result to return. Can be 'list' or 
            'function'. If 'list', a list of points is returned. If 'function',
            a lambda function for the trajectory is returned.

    """
    # Points
    P0 = np.array(point_0.pos)
    P1 = np.array(point_1.pos)

    # Direction vectors
    D0 = np.array(point_0.dir)
    D1 = -np.array(point_1.dir)

    # Control points (adjust the t parameter as needed)
    t = 0.5
    C0 = P0 + t * D0
    C1 = P1 + t * D1

    # Generate the curve
    t_values = np.linspace(0, 1, 100)
    curve_np = (1-t_values)**3 * P0[:,None] + \
            3*(1-t_values)**2 * t_values * C0[:,None] + \
            3*(1-t_values) * t_values**2 * C1[:,None] + \
            t_values**3 * P1[:,None]
    
    curve = []
    for i in range(len(curve_np[0])):
        curve.append((curve_np[0][i], curve_np[1][i]))
    return curve

def build_trajectory_from_waypoint(start_waypoint, rng=None):
    """
    Build a continuous trajectory starting from 'start_waypoint' by following the 
    connection information. If a waypoint has multiple connections, a random choice
    is made for the next waypoint using the provided random generator for replicability.
    
    Args:
        start_waypoint (WayPoint): The starting waypoint.
        rng (random.Random, optional): Random generator instance. If None, use the global random module.
    
    Returns:
        traj (list): A list of points forming the trajectory.
    """
    traj = []
    current_wp = start_waypoint
    visited = set()

    while current_wp.next_waypoints:
        # Avoid cycles.
        if id(current_wp) in visited:
            break
        visited.add(id(current_wp))
        
        # Randomly choose the next waypoint using the provided generator.
        if len(current_wp.next_waypoints) > 1:
            if rng is None:
                next_wp = random.choice(current_wp.next_waypoints)
            else:
                next_wp = rng.choice(current_wp.next_waypoints)
        else:
            next_wp = current_wp.next_waypoints[0]
        
        segment = calc_traj(current_wp, next_wp)
        # Avoid duplicate points between segments.
        if traj:
            segment = segment[1:]
        traj.extend(segment)
        current_wp = next_wp

    return traj

class Joy():
    """
    Joystick class that takes input from a joystick and return the speed and
    steering angle of the vehicle.
    """
    def __init__(self, max_speed=0.5, max_steering_angle=15, verbose=False):
        """
        Args:
            max_speed (float): maximum speed of the vehicle
            max_steering_angle (float): maximum steering angle of the vehicle
            verbose (boolean): print the input values
        """
        self.max_speed = max_speed
        self.max_steering_angle = max_steering_angle
        self.verbose = verbose
        self.speed = 0
        self.steering_angle = 0

        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0) 
            self.joystick.init()
            print(f"Joystick {self.joystick.get_name()} initialized.")
        else:
            print("No joystick found.")
            pygame.quit()
            exit()

    def get_input(self):
        """
        Get the input from the joystick and return the speed and steering angle
        of the vehicle.

        Returns:
            speed (float): speed of the vehicle
            steering_angle (float): steering angle of the vehicle
        """

        joy4 = self.joystick.get_axis(4)
        joy0 = self.joystick.get_axis(0)
        if abs(joy4) < 0.1:
            joy4 = 0
        if abs(joy0) < 0.1:
            joy0 = 0
        self.speed = - joy4 * self.max_speed
        self.steering_angle = - joy0 * self.max_steering_angle
        if self.verbose:
            print(f"Speed: {self.speed}, Steering Angle: {self.steering_angle}")
        return (self.speed, np.radians(self.steering_angle))


# ===========================================
# Author: Dongshuai Su
# Date: 02/26/2026
# ===========================================
def load_config_traf_lights(file):
    """Unpack JSON config file and return configurations for traffic lights
    
        Args: 
            file: traffic lights' JSON config file
    """

    # Load JSON file
    with open(file, 'r') as f:
        config = json.load(f)

    # List for objects
    # If do instantiations in this function, uncomment
    traf_lights = []
    
    for conf in config["traffic_lights"]:
        pos = tuple(conf["pos"])
        red_time = conf.get("red_time", 3)
        yel_time = conf.get("yel_time", 3)
        gre_time = conf.get("gre_time", 3)
        initial_state = conf.get("initial_state", "RED")
        initial_delay = conf.get("initial_delay", 0)
        traf_lights.append(TrafficLight(pos, red_time, gre_time, yel_time, initial_state, initial_delay))
        
        # Undecided if do instantiations internally
        # Mock code:
        # traf_light_obj = TrafficLight(........)
        # append traf_light_obj into traf_lights

    # If do instantiations internnaly, return objects
    # If not, return config dic/lists 

    # Currently mock return
    return traf_lights

# ===========================================
# Author: James Zhang
# Date: 03/03/2025
# ===========================================
    
class TrafficLight:
    def __init__(self, pos, red_time, green_time, yellow_time, initial_state='RED', wait=0):
        self.red_time = red_time
        self.green_time = green_time
        self.yellow_time = yellow_time
        self.wait = wait
        self.curr = initial_state
        self.time_elapsed = 0
        self.pos = pos
    def update(self, time_step=1):
        self.time_elapsed += time_step
        if self.curr == 'RED':
            if self.wait > 0:
                if self.time_elapsed >= self.wait:
                    self.curr = 'GREEN'
                    self.time_elapsed = 0
                    self.wait = 0
            elif self.time_elapsed >= self.red_time:
                self.curr = 'GREEN'
                self.time_elapsed = 0
        elif self.curr == 'GREEN':
            if self.time_elapsed >= self.green_time:
                self.curr = 'YELLOW'
                self.time_elapsed = 0
        elif self.curr == 'YELLOW':
            if self.time_elapsed >= self.yellow_time:
                self.curr = 'RED'
                self.time_elapsed = 0
        return self
    def get_pos(self):
        return self.pos
    def get_state(self):
        return self.curr
        