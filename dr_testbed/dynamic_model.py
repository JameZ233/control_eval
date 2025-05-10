import numpy as np
from abc import ABC, abstractmethod

class BasicDynamicModel(ABC):
    """
    Base class for all dynamic models. A dynamic model is responsible for
    updating the state of the vehicle base on the input action.
    """
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, state, velocities, steerings, dt):
        """
        Update the state of the vehicle base on the input action.
        Args:
            state (tuple): state of the vehicle. The state is a tuple of 
                ((x, y), yaw), where (x, y) is the position of the vehicle and
                yaw is the heading of the vehicle.
            velocities (float): velocity of the vehicle
            steerings (float): steering angel of the vehicle
            dt (float): time step
        """
        pass

    @abstractmethod
    def future_traj(self, state, velocities, steerings, dts):
        """
        Generate the future trajectory of the vehicle.
        Args:
            state (tuple): state of the vehicle. The state is a tuple of 
                ((x, y), yaw), where (x, y) is the position of the vehicle and
                yaw is the heading of the vehicle.
            velocities (list): list of velocities of the vehicle
            steerings (list): list of steering angles of the vehicle
            dts (list): list of time steps
        """
        pass

class KinematicBicycle(BasicDynamicModel):
    """
    Kinematic Bicycle model that takes velocity and steering angel as input
    and return the new position and heading of the vehicle.
    """
    def __init__(self,
                 wheel_base = 0.15,
                 max_steering_angel = np.radians(15),
                 velocity_range = [-0.2, 0.5]
                 ):
        """
        Args:
            wheel_base (float): distance between the front and rear wheel.
            max_steering_angel (float): maximum steering angel of the vehicle.
            velocity_range (list): range of velocity of the vehicle.
        """
        
        self.wheel_base = wheel_base
        self.max_steering_angel = max_steering_angel
        self.velocity_range = velocity_range

    def __call__(self,
                 state,
                 velocity,
                 steering,
                 dt):
        """
        Update the position and heading of the vehicle base on the input
        velocity and steering.
        Args:
            state (tuple): state of the vehicle. The state is a tuple of 
                ((x, y), yaw), where (x, y) is the position of the vehicle and
                yaw is the heading of the vehicle.
            velocity (float): velocity of the vehicle
            steering (float): steering angel of the vehicle
            dt (float): time step
        """
        assert len(state) == 3
        x, y, yaw = state

        assert velocity >= self.velocity_range[0] \
            and velocity <= self.velocity_range[1]
        
        assert steering <= self.max_steering_angel \
            and steering >= -self.max_steering_angel

        yaw_rate = velocity * np.tan(steering) / self.wheel_base
        new_yaw = yaw + yaw_rate * dt
        avg_yaw = 0.5 * (yaw + new_yaw)
        x += velocity * np.cos(avg_yaw) * dt
        y += velocity * np.sin(avg_yaw) * dt
        yaw = (new_yaw + np.pi) % (2 * np.pi) - np.pi

        return np.array([x, y, yaw])
    
    def future_traj(self,
                    state,
                    velocities,
                    steerings,
                    dts):
        """
        Generate the future trajectory of the vehicle.
        Args:
            state (tuple): state of the vehicle. The state is a tuple of 
                ((x, y), yaw), where (x, y) is the position of the vehicle and
                yaw is the heading of the vehicle.
            velocities (list): list of velocities of the vehicle
            steerings (list): list of steering angles of the vehicle
            dts (list): list of time steps
        """
        assert len(velocities) == len(steerings) == len(dts)
        traj = []
        for i in range(len(velocities)):
            state = self(state, velocities[i], steerings[i], dts[i])
            traj.append(state)
        return np.array(traj)