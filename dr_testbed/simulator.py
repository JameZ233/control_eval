import numpy as np
from .vehicle import Vehicle
from .utility import TrafficLight
import math

class Simulator:
    """
    A lightweight simulator for the deepracer testbed.
    """

    def __init__(self,
                 num_vehicle=1,
                 lights=None,       # now expect a list of TrafficLight objects
                 vehicle_states=None,
                 map_layout='map_1',
                 map_size=[6.0, 6.0],
                 time_step=1/60,
                 ):
        """
        Args:
            num_vehicle (int): number of vehicles controlled through commands received from outside.
            lights (list): list of TrafficLight objects.
            vehicle_states (list): initial states for vehicles.
            map_layout (str): map layout identifier.
            map_size (list): dimensions of the map.
            time_step (float): simulation time step.
        """

        self.num_vehicle = num_vehicle
        self.map_layout = map_layout
        self.map_size = map_size
        self.time_step = time_step
        # Expect lights to be a list of TrafficLight objects or None.
        self.lights = lights if lights is not None else []

        self.vehicle_size = [0.3 / map_size[0] * 800, 0.2 / map_size[0] * 800]

        self.vehicle = []
        if vehicle_states is not None:
            for state in vehicle_states:
                self.vehicle.append(Vehicle(*state))
        for i in range(num_vehicle):
            self.vehicle.append(Vehicle())

        # Directly assign the list of TrafficLight objects
        self.traffic_lights = self.lights

    def step(self, vehicle_actions):
        """
        Update the state of the simulation based on the input actions.

        Args:
            vehicle_actions (list): list of actions for each vehicle. 
                                     Each action is a tuple of (velocity, steering).

        Returns:
            vehicle_states (list): list of states for each vehicle. Each state is a tuple of (x, y, yaw).
            traffic_lights_info (list): list of tuples containing the position and state of each traffic light.
        """

        assert len(vehicle_actions) == self.num_vehicle, \
            "Number of actions should match number of vehicles."

        vehicle_states = []

        for i in range(self.num_vehicle):
            pos = self.vehicle[i].step(*vehicle_actions[i], self.time_step)
            vehicle_states.append(pos)

        # Update each traffic light and get its position and state.
        for tl in self.traffic_lights:
            tl.update(self.time_step)
        return vehicle_states, self.traffic_lights


        
