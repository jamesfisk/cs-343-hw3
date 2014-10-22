from __future__ import division
from OpenNero import *
from common import *
import random
import math

import Maze
from Maze.constants import *
import Maze.agent
from Maze.agent import *

class MyTabularRLAgent(AgentBrain):
    """
    Tabular RL Agent
    """
    def __init__(self, gamma, alpha, epsilon):
        """
        Constructor that is called from the robot XML file.
        Parameters:
        @param gamma reward discount factor (between 0 and 1)
        @param alpha learning rate (between 0 and 1)
        @param epsilon parameter for the epsilon-greedy policy (between 0 and 1)
        """
        AgentBrain.__init__(self) # initialize the superclass
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        """
        Our Q-function table. Maps from a tuple of observations (state) to 
        another map of actions to Q-values. To look up a Q-value, call the predict method.
        """
        self.Q = {} # our Q-function table
        print 
    
    def __str__(self):
        return self.__class__.__name__ + \
            ' with gamma: %g, alpha: %g, epsilon: %g' \
            % (gamma, alpha, epsilon)
    
    def initialize(self, init_info):
        """
        Create a new agent using the init_info sent by the environment
        """
        self.action_info = init_info.actions
        self.sensor_info = init_info.sensors
        return True
    
    def predict(self, observations, action):
        """
        Look up the Q-value for the given state (observations), action pair.
        """
        o = tuple([x for x in observations])
        if o not in self.Q:
            return 0
        else:
            return self.Q[o][action]
    
    def update(self, observations, action, new_value):
        """
        Update the Q-function table with the new value for the (state, action) pair
        and update the blocks drawing.
        """
        o = tuple([x for x in observations])
        actions = self.get_possible_actions(observations)  #get actions does not use observations
        if o not in self.Q:
            self.Q[o] = [0 for a in actions]
        self.Q[o][action] = new_value
        self.draw_q(o)
    
    def draw_q(self, o):
        e = get_environment()
        if hasattr(e, 'draw_q'):
            e.draw_q(o, self.Q)

    def get_possible_actions(self, observations):
        """
        Get the possible actions that can be taken given the state (observations)
        """
        aMin = self.action_info.min(0)
        aMax = self.action_info.max(0)
        actions = range(int(aMin), int(aMax+1))
        return actions
        
    def get_max_action(self, observations):
        """
        get the action that is currently estimated to produce the highest Q
        """
        actions = self.get_possible_actions(observations)
        max_action = actions[0]
        max_value = self.predict(observations, max_action)
        for a in actions[1:]:
            value = self.predict(observations, a)
            if value > max_value:
                max_value = value
                max_action = a
        return (max_action, max_value)

    def get_epsilon_greedy(self, observations, max_action = None, max_value = None):
        """
        get the epsilon-greedy action
        """
        actions = self.get_possible_actions(observations)
        if random.random() < self.epsilon: # epsilon of the time, act randomly
            return random.choice(actions)
        elif max_action is not None and max_value is not None:
            # we already know the max action
            return max_action
        else:
            # we need to get the max action
            (max_action, max_value) = self.get_max_action(observations)
            return max_action
    
    def start(self, time, observations):
        """
        Called to figure out the first action given the first observations
        @param time current time
        @param observations a DoubleVector of observations for the agent (use len() and [])
        """
        self.previous_observations = observations
        self.previous_action = self.get_epsilon_greedy(observations)
        return self.previous_action

    def act(self, time, observations, reward):
        """
        return an action given the reward for the previous action and the new observations
        @param time current time
        @param observations a DoubleVector of observations for the agent (use len() and [])
        @param the reward for the agent
        """
        # get the reward from the previous action
        r = reward[0]
        
        # get the updated epsilon, in case the slider was changed by the user
        self.epsilon = get_environment().epsilon
        
        # get the old Q value
        Q_old = self.predict(self.previous_observations, self.previous_action)
        
        # get the max expected value for our possible actions
        (max_action, max_value) = self.get_max_action(observations)
        
        # update the Q value
        self.update( \
            self.previous_observations, \
            self.previous_action, \
            Q_old + self.alpha * (r + self.gamma * max_value - Q_old) )
        
        # select the action to take
        action = self.get_epsilon_greedy(observations, max_action, max_value)
        self.previous_observations = observations
        self.previous_action = action
        return action

    def end(self, time, reward):
        """
        receive the reward for the last observation
        """
        # get the reward from the last action
        r = reward[0]
        o = self.previous_observations
        a = self.previous_action

        # get the updated epsilon, in case the slider was changed by the user
        self.epsilon = get_environment().epsilon

        # Update the Q value
        Q_old = self.predict(o, a)
        q = self.update(o, a, Q_old + self.alpha * (r - Q_old) )
        return True

class MyTilingRLAgent(MyTabularRLAgent):
    """
    Tiling RL Agent
    """
    def __init__(self, gamma, alpha, epsilon):
        """
        Constructor that is called from the robot XML file.
        Parameters:
        @param gamma reward discount factor (between 0 and 1)
        @param alpha learning rate (between 0 and 1)
        @param epsilon parameter for the epsilon-greedy policy (between 0 and 1)
        """
        MyTabularRLAgent.__init__(self, gamma, alpha, epsilon) # initialize the superclass
        for i in range(8):
          for j in range(8):
             self.Q[(i, j)] = 0

    def update(self, observations, action, new_value):
        """
        Update the Q-function table with the new value for the (state, action) pair
        and update the blocks drawing.  Update to only insert 8x8 tile environment.
        Convert the observation position to the 8x8 tile position.
        """
        o = tuple([x for x in observations])
        #convert x,y position to tile position

        #combine action taken with observations to get destination x,y
        #convert destination x,y to row,col coarse loc
        fine_loc = [o[0], o[1]]

        if (action == 0):
            fine_loc[0] += 1
        elif (action == 1):
            fine_loc[0] -= 1
        elif (action == 2):
            fine_loc[1] += 1
        else:
            fine_loc[1] -= 1

        coarse_loc = get_environment().maze.xy2rc(fine_loc[0], fine_loc[1])
        actions = self.get_possible_actions(observations)
        
        self.Q[coarse_loc] = new_value

    def predict(self, observations, action):
        """
        Look up the Q-value for the given state (observations), action pair.
        Need to look up the value of the 8x8 tile the position is currently
        a part of. 
        """
        o = tuple([x for x in observations])
        
        coarse_loc = self.fine2coarse(observations, action)     

        initial_coarse = get_environment().maze.xy2rc(o[0], o[1])       

        if ((coarse_loc[0], coarse_loc[1]), (initial_coarse[0], initial_coarse[1])) in get_environment().maze.walls:
            return -1
        #convert observations to 8X8 position
        if coarse_loc not in self.Q:
            return 0
        else:
            return self.Q[coarse_loc]

    def get_possible_actions(self, observations):
        """
        Get the possible actions that can be taken given the state (observations)
        """
        o = tuple([x for x in observations])
        aMin = self.action_info.min(0)
        aMax = self.action_info.max(0)
        actions = range(int(aMin), int(aMax+1))
        possible_actions = []
        for action in actions:
            coarse_loc = self.fine2coarse(observations, action)
            initial_coarse = get_environment().maze.xy2rc(o[0], o[1])       
            if ((coarse_loc[0], coarse_loc[1]), (initial_coarse[0], initial_coarse[1])) not in get_environment().maze.walls:
                possible_actions.append((action, observations[action + 2]))
        
        possible_actions.sort(key=lambda tup : tup[1]) 
       
        output_actions = [x[0] for x in possible_actions]
        output_actions.reverse()
        return output_actions
        
    def get_max_action(self, observations):
        """
        get the action that is currently estimated to produce the highest Q
        """
        actions = self.get_possible_actions(observations)
        max_action = actions[0]
        max_value = self.predict(observations, max_action)
        for a in actions[1:]:
            value = self.predict(observations, a)
            if value > max_value:
                max_value = value
                max_action = a
        return (max_action, max_value)

    def fine2coarse(self, observations, action):
        o = tuple([x for x in observations])
        
        fine_loc = [o[0], o[1]]

        if (action == 0):
            fine_loc[0] += 1
        elif (action == 1):
            fine_loc[0] -= 1
        elif (action == 2):
            fine_loc[1] += 1
        else:
            fine_loc[1] -= 1

        coarse_loc = get_environment().maze.xy2rc(fine_loc[0], fine_loc[1]) 
        return coarse_loc


class MyNearestNeighborsRLAgent(MyTabularRLAgent):
    """
    Nearest Neighbors RL Agent
    """
    def __init__(self, gamma, alpha, epsilon):
        """
        Constructor that is called from the robot XML file.
        Parameters:
        @param gamma reward discount factor (between 0 and 1)
        @param alpha learning rate (between 0 and 1)
        @param epsilon parameter for the epsilon-greedy policy (between 0 and 1)
        """
        MyTabularRLAgent.__init__(self, gamma, alpha, epsilon) # initialize the superclass


    def update(self, observations, action, r, max_value):
        """
        Update the Q-function table with the new value for the (state, action) pair
        and update the blocks drawing.  Update to only insert 8x8 tile environment.
        Convert the observation position to the 8x8 tile position.
        """
        o = tuple([x for x in observations])
        #convert x,y position to tile position
        coarse_loc = get_environment().maze.xy2rc(o[0], o[1])

        neighbors = self.neighbors(observations, action)
        distances = self.distance(observations, action, neighbors)
        weights = self.weight(observations, distances)

        for item in weights:
            coarse_loc1 = item[0]
            if (coarse_loc1 not in self.Q):
                self.Q[coarse_loc1] = 0
            self.Q[coarse_loc1] = self.Q[coarse_loc1] + self.alpha * item[1] * (r + self.gamma * max_value - self.Q[coarse_loc1])

    
    def end(self, time, reward):
        """
        Receive the reward for the last observation
        """
        # get the reward from the last action
        r = reward[0]
        o = self.previous_observations
        a = self.previous_action

        # get the updated epsilon, in case the slider was changed by the user
        self.epsilon = get_environment().epsilon

        # Update the Q value
        q = self.update(o, a, r, 0)
        return True

    def act(self, time, observations, reward):
        """
        return an action given the reward for the previous action and the new observations
        @param time current time
        @param observations a DoubleVector of observations for the agent (use len() and [])
        @param the reward for the agent
        """
        # get the reward from the previous action
        r = reward[0]
        
        # get the updated epsilon, in case the slider was changed by the user
        self.epsilon = get_environment().epsilon
        
        # get the max expected value for our possible actions
        (max_action, max_value) = self.get_max_action(observations)
        
        # update the Q value
        self.update( \
            self.previous_observations, \
            self.previous_action, \
            r, max_value)
        
        # select the action to take
        action = self.get_epsilon_greedy(observations, max_action, max_value)
        self.previous_observations = observations
        self.previous_action = action
        return action

    def get_max_action(self, observations):
        """
        Get the action that is currently estimated to produce the highest Q
        """
        actions = self.get_possible_actions(observations)
        max_action = actions[0]
        neighbors = self.neighbors(observations, actions[0])
        distances = self.distance(observations, actions[0], neighbors)
        weights = self.weight(observations, distances)
        
        max_value = 0 
        for item in weights:
            if item[0] not in self.Q:
                self.Q[item[0]] = 0
            max_value += self.Q[item[0]] * item[1]
        
        for a in actions[1:]:
            neighbors = self.neighbors(observations, a)
            distances = self.distance(observations, a, neighbors)
            weights = self.weight(observations, distances)
            value = 0
            for item in weights:
                if item[0] not in self.Q:
                    self.Q[item[0]] = 0
                value += self.Q[item[0]] * item[1]
            if value > max_value:
                max_value = value
                max_action = a
        return (max_action, max_value)


    def get_possible_actions(self, observations):
        """
        Get the possible actions that can be taken given the state (observations)
        """
        o = tuple([x for x in observations])
        aMin = self.action_info.min(0)
        aMax = self.action_info.max(0)
        actions = range(int(aMin), int(aMax+1))
        possible_actions = []
        for action in actions:
            coarse_loc = self.fine2coarse(observations, action)
            initial_coarse = get_environment().maze.xy2rc(o[0], o[1])       
            if ( ((coarse_loc[0], coarse_loc[1]), (initial_coarse[0], initial_coarse[1])) not in get_environment().maze.walls and
                    coarse_loc[0] <= 7 and coarse_loc[0] >= 0 and coarse_loc[1] >= 0 and coarse_loc[1] <= 7):
                
                #Save space and time by accounting for nearby walls
                if ( observations[action + 2] > .7):
                    possible_actions.append(action)
        
        return possible_actions


    def get_epsilon_greedy(self, observations, max_action = None, max_value = None):
        """
        Get the epsilon-greedy action
        """
        actions = self.get_possible_actions(observations)
        if random.random() < self.epsilon: # epsilon of the time, act randomly
            return random.choice(actions)
        elif max_action is not None and max_value is not None:
            # we already know the max action
            return max_action
        else:
            # we need to get the max action
            (max_action, max_value) = self.get_max_action(observations)
            return max_action

    def predict(self, observations, action):
        """
        Look up the Q-value for the given state (observations), action pair.
        """
        o = tuple([x for x in observations])
        #convert x,y to tile position
        coarse_loc = get_environment().maze.xy2rc(o[0], o[1])

        #convert observations to 8X8 position
        if coarse_loc not in self.Q:
            return 0
        else:
            return self.Q[coarse_loc]

    def fine2coarse(self, observations, action):
				"""
				Get the coarse grained location after a fine grained action
				"""
        o = tuple([x for x in observations])
        
        fine_loc = [o[0], o[1]]

        if (action == 0):
            fine_loc[0] += 1
        elif (action == 1):
            fine_loc[0] -= 1
        elif (action == 2):
            fine_loc[1] += 1
        else:
            fine_loc[1] -= 1

        coarse_loc = get_environment().maze.xy2rc(fine_loc[0], fine_loc[1]) 
        return coarse_loc

    def fine_loc_after_action(self, observations, action):
				"""
		    Get the fine grained location of agent after a specified action
				"""
        o = tuple([x for x in observations])
        
        fine_loc = [o[0], o[1]]

        if (action == 0):
            fine_loc[0] += 1
        elif (action == 1):
            fine_loc[0] -= 1
        elif (action == 2):
            fine_loc[1] += 1
        else:
            fine_loc[1] -= 1
        return tuple(fine_loc)



    def weight (self, observations, distances):
			 """
			 Calculate the weight of each move given distance
			 """
       sum_all = 0
       weights = []
       for item in distances:
           sum_all += item[1]
       for item in distances:
           weights.append((item[0], 1.0 - (item[1] / sum_all)))
       weights.sort(key=lambda tup : tup[1])
       return weights


    def distance (self, observations, action, neighbors):
       """
       Calculate distances to different tiles
       """

       o = tuple([x for x in observations])
       neighbor_distance = []
       for item in neighbors:
           fine_loc_of_big_tile = get_environment().maze.rc2xy(item[0], item[1])
           my_fine_loc = self.fine_loc_after_action(observations, action)
           distance = math.sqrt((fine_loc_of_big_tile[0] - my_fine_loc[0]) ** 2 + (fine_loc_of_big_tile[1] - my_fine_loc[1]) ** 2)
           neighbor_distance.append((item, distance))

       neighbor_distance.sort(key=lambda tup : tup[1]) 
       return neighbor_distance[:3]
       

    def neighbors (self, observations, action):
        """
        Return valid neighboring tiles
        """
        o = tuple([x for x in observations])
        #convert x,y position to tile position
        coarse_loc = self.fine2coarse(observations, action)
        r1 = coarse_loc[0]
        c1 = coarse_loc[1]
        neighbors = []
        for r in range(-1, 2):
            for c in range(-1, 2):
                #Check position is in maze
                if ((r1 + r >= 0) and (r1 + r <= 7) and (c1 + c >= 0) and (c1 + c <= 7)):
                    #No simple walls
                    if not ((r1, c1), (r1 + r, c1 + c)) in get_environment().maze.walls:   
                         #No complex walls
                         if (r != 0 and c != 0):     #Check for diagonals
                             if ((((r1, c1), (r1 + r, c1)) not in get_environment().maze.walls and
                                ((r1 + r, c1), (r1 + r, c1 + c)) not in get_environment().maze.walls) or
                                (((r1, c1), (r1, c1 + c)) not in get_environment().maze.walls and
                                ((r1, c1 + c), (r1 + r, c1 + c)) not in get_environment().maze.walls)):
                                    neighbors.append((r1 + r, c1 + c))
                         else:
                             neighbors.append((r1 + r, c1 + c))
        return neighbors

