import sys
import os
import numpy as np
from game import Directions

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.qlearning_agent import FinalQAgent

#Storage
WEIGHTS_FILE = "q_learning_weights.npy"
EPISODE_FILE = "q_learning_episode.txt"

def load_episode_number():
    try:
        if os.path.exists(EPISODE_FILE):
            with open(EPISODE_FILE, 'r') as f:
                return int(f.read().strip())
    except:
        pass
    return 0

def save_episode_number(episode):
    try:
        with open(EPISODE_FILE, 'w') as f:
            f.write(str(episode))
    except:
        pass

def load_weights():
    try:
        if os.path.exists(WEIGHTS_FILE):
            return np.load(WEIGHTS_FILE)
    except:
        pass
    return None

def save_weights(weights):
    try:
        np.save(WEIGHTS_FILE, weights)
    except:
        pass

class TrainAgentFixed:
    def __init__(self, alpha=0.05, gamma=0.9, epsilon=0.5):
        #Convert string args to floats
        alpha = float(alpha) if isinstance(alpha, str) else alpha
        gamma = float(gamma) if isinstance(gamma, str) else gamma
        epsilon = float(epsilon) if isinstance(epsilon, str) else epsilon
        
        self.episode_num = load_episode_number() + 1
        save_episode_number(self.episode_num)
        
        #Decay epsilon
        decayed_epsilon = epsilon * (0.85 ** (self.episode_num - 1))
        decayed_epsilon = max(0.05, decayed_epsilon)
        
        #Create agent
        self.agent = FinalQAgent(alpha, gamma, decayed_epsilon)
        
        self.last_state = None
        self.last_action = None
        self.total_reward = 0
    
    def getAction(self, state):
        action = self.agent.getAction(state)
        
        #Learn from previous step
        if self.last_state is not None and self.last_action is not None:
            reward = self.agent.get_reward(self.last_state, self.last_action, state)
            self.agent.update(self.last_state, self.last_action, reward, state)
            self.total_reward += reward
        
        self.last_state = state
        self.last_action = action
        return action
    
    def final(self, state):
        #Final learning step
        if self.last_state is not None and self.last_action is not None:
            reward = self.agent.get_reward(self.last_state, self.last_action, state)
            self.agent.update(self.last_state, self.last_action, reward, state)
            self.total_reward += reward
        
        #Save weights
        save_weights(self.agent.weights)
        
        #Save episode completion
        save_episode_number(self.episode_num)
        
        return Directions.STOP