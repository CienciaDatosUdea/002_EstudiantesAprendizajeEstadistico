# ml/final_q_agent.py
import numpy as np
from game import Agent
from pacman import Directions

class FinalQAgent(Agent):
    def __init__(self, alpha=0.05, gamma=0.9, epsilon=0.5):
        super().__init__()
        #Convert string args to floats
        if isinstance(alpha, str): alpha = float(alpha)
        if isinstance(gamma, str): gamma = float(gamma)
        if isinstance(epsilon, str): epsilon = float(epsilon)
        
        self.alpha = alpha      #Learning rate
        self.gamma = gamma      #Discount factor
        self.epsilon = epsilon  ##"random" rate
        self.weights = np.zeros(3)  #3 features
    
    def get_features(self, state, action):
        
        pac_x, pac_y = state.getPacmanPosition()
        
        #Movement
        if action == Directions.NORTH: pac_y += 1
        elif action == Directions.SOUTH: pac_y -= 1
        elif action == Directions.EAST: pac_x += 1
        elif action == Directions.WEST: pac_x -= 1
        
        # Game state
        food = state.getFood().asList()
        ghosts = state.getGhostPositions()
        
        #Food? (y: 1, n:0)
        will_eat = 1.0 if (pac_x, pac_y) in food else 0.0
        
        #Close to food? (y: 1, n:0)
        if food:
            distances = [abs(pac_x - x) + abs(pac_y - y) for x, y in food]
            closest = min(distances)
            food_score = 1.0 / (1.0 + closest)  # 1.0 if on food, smaller if far
        else:
            food_score = 0.0
        
        #Danger? (y: 1, n:0)
        danger = 0.0
        for gx, gy in ghosts:
            dist = abs(pac_x - gx) + abs(pac_y - gy)
            if dist == 0:  # Would collide with ghost
                danger = 1.0
            elif dist == 1:  # Very close to ghost
                danger = 0.7
            elif dist == 2:  # Somewhat close
                danger = 0.3
        
        return np.array([will_eat, food_score, danger])
    
    def getQValue(self, state, action):
        #Q(s,a) = weights • features
        features = self.get_features(state, action)
        return np.dot(self.weights, features)
    
    def getAction(self, state):
        legal = state.getLegalPacmanActions()
        if not legal:
            return Directions.STOP
        
        if np.random.rand() < self.epsilon:
            return np.random.choice(legal)
        
        q_values = [self.getQValue(state, a) for a in legal]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(legal, q_values) if q == max_q]
        return np.random.choice(best_actions)
    
    def update(self, state, action, reward, next_state):
        #Best Q-value for next state
        legal_next = next_state.getLegalPacmanActions()
        if legal_next:
            next_q = max(self.getQValue(next_state, a) for a in legal_next)
        else:
            #For reset
            next_q = 0
        
        #TD error
        current_q = self.getQValue(state, action)
        td_error = reward + self.gamma * next_q - current_q
        
        if reward > 0:
            learn_rate = self.alpha * 1.6
        else:
            learn_rate = self.alpha * 0.7
        
        #Get features
        features = self.get_features(state, action)
        
        #Update weights
        self.weights += learn_rate * td_error * features
        
        for i in range(len(self.weights)):
            if self.weights[i] > 15:
                self.weights[i] = 15 + (self.weights[i] - 15) * 0.9  #Slow growth
            elif self.weights[i] < -15:
                self.weights[i] = -15 + (self.weights[i] + 15) * 0.9  #Slow decay

    def get_reward(self, state, action, next_state):
        reward = 0
        
        # :D
        # Survival
        reward += 2
        
        # 2. Food
        score_diff = next_state.getScore() - state.getScore()
        if score_diff == 10:
            reward += 15  
        
        # 3. Ghost eating
        elif score_diff == 200: 
            reward += 250 
        
        # Wiiin
        if next_state.isWin():
            return 500 
        
        #:((
        if next_state.isLose():
            return -100 

        pacman_pos = state.getPacmanPosition()
        
        #Bonus for being near food 
        food = state.getFood().asList()
        if food:
            dists = [abs(pacman_pos[0] - x) + abs(pacman_pos[1] - y) for x, y in food]
            if min(dists) <= 2:  
                reward += 3
        
        #Penalty for being near ghost
        for ghost in state.getGhostStates():
            if ghost.scaredTimer == 0: 
                gx, gy = ghost.getPosition()
                dist = abs(pacman_pos[0] - gx) + abs(pacman_pos[1] - gy)
                if dist == 1:
                    reward -= 10
                elif dist == 2:
                    reward -= 3
        
        return reward