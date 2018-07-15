import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
#        self.init_pose = init_pose
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 5

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 1000
        self.action_size = 4

        # Goal
        # tj: why does this not involve the Euler angles, only the positional coordinates?
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 20.]) 

    def get_reward(self ):
        """Uses current pose of sim to return reward."""
#        xy_velocity_penalty = np.abs( self.sim.v[0] ) + np.abs( self.sim.v[1] ) # only x and y velocities are penalized
#        ang_velocity_penalty = abs(self.sim.angular_v).sum() # O(0.1))
        xy_deviation2 = np.sqrt( np.sum([ d**2 for d in (self.sim.pose[:2] - self.target_pos[:2])]) )
#        total_L2_deviation = np.sum([ d**2 for d in (self.sim.pose[:3] - self.target_pos)]) #O(1000)
        total_L1_deviation = (abs(self.sim.pose[:3] - self.target_pos)).sum() #O(1000)
        z_velocity = self.sim.v[2]
        z = self.sim.pose[2]    

        
        reward_target_z_nbd = 2 if abs(z - self.target_pos[2])<1 else 0
        reward_target_xy_nbd = 10 if ( reward_target_z_nbd==1 and xy_deviation2<2 ) else 0
        reward_l1 = -5e-3 * total_L1_deviation #the farther you are from target, the more negative it is
#        reward_l2 = -5e-3 * total_L2_deviation #the farther you are from target, the more negative it is
#        reward_z_velocity = 1e-2 * ( self.target_pos[2] -z ) * z_velocity #reward z velocity if below target and penalize if above target 
#        reward_xydeviation = -1e-2*xy_deviation2 #penalize x and y deviations in the 
        # The reward below should cause a slowdown as the copter approaches the target 
        reward_z_between_terminals = 1e-4 * (self.target_pos[2]-z) * (z-10) #if (self.init_pose[2]<z<self.target_pos[2]) else 0#positive between start and end and negative otherwise        
#        reward_sqrt_l1 = -1e-3 * np.sqrt( total_L1_deviation )
#        reward_squared_l1 = -1e-4 * ( total_L1_deviation**2 )
        
#        reward_target_nbd = 10 if total_L1_deviation<2 else 0 
#        reward_target_vicinity = 1e-2 / (0.1 + total_L2_deviation )
#       
#        reward_list = [1, reward_z_velocity, reward_target_nbd]
#        reward_list = [1, reward_l1, reward_z_velocity, reward_target_nbd] # this learned
        reward_list = [0.5, reward_l1, reward_z_between_terminals, reward_target_z_nbd, reward_target_xy_nbd]

#        reward_list = [0.1, reward_l1, reward_z_velocity, reward_target_nbd]
        
#        reward_list = [0.1, reward_l1, reward_xydeviation, reward_target_nbd, reward_z_between_terminals ] #, reward_xydeviation, reward_z_between_terminals ] 
        reward = np.sum(reward_list)
        reward = np.tanh(reward)
        
        
#        if ( 5 > self.sim.pose[2] > 4.4 ):
#            print('xy_v_penalty=', xy_velocity_penalty, ' omega_penalty=', ang_velocity_penalty, 
#              ' xy_deviation2=',xy_deviation2, ' total_L2_deviation=', total_L2_deviation, ' total_L1_deviation=',total_L2_deviation)
#            return   
#        print('reward_l1=', reward_l1, ' reward_sqrt_l1=', reward_sqrt_l1, 
#              ' reward_squared_l1=',reward_squared_l1, ' reward_target_vicinity=', reward_target_vicinity)
        
        
#        reward = np.tanh(reward)#        
        
#        reward = 0
#        reward += reward_l1
        # adding both square and sqrt keep driving to the target strongly (sqrt is weak when far and square is weak when near)
#        reward += reward_sqrt_l1
#        reward += reward_squared_l1
#        if total_L2_deviation < 2:
        # reward for being close to the target
#        reward += reward_target_vicinity
        # reward z-velocity if you are getting close to the ground
#        if self.sim.v[2] < 5:
#            reward += 0.5 * z_velocity / (0.01+self.sim.pose[2])
        
#        reward -= 0.01 * xy_velocity_penalty
        
        # try an exponential reward
#        reward = np.exp( -0.1*total_L1_deviation ) #this is like the potential described in the Udacity RL course (Isbell & Littman)
#        reward += 0.1 * z_velocity**2
#        print(rotor_speeds) How to penalize vastly different rotor speeds if
#        reward -= 0.1 * (np.max(rotor_speeds) - np.min(rotor_speeds) )
#        reward -= 0.01 * ang_velocity_penalty
        
#        x_penalty = .1
#        y_penalty = .1
#        z_penalty = 1.5
#        reward =  1. - x_penalty*abs(self.sim.pose[0]-self.target_pos[0]) -y_penalty*abs(self.sim.pose[1]-self.target_pos[1]) - z_penalty*abs(self.sim.pose[2]-self.target_pos[2])
#        reward = np.tanh( reward )
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state