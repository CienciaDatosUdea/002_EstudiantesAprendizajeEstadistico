from game import Agent
from game import Directions
import random

class RandomAgent(Agent):
    def __init__(self):
        super().__init__()
    
    def getAction(self, state):
        #Get legal actions
        legal = state.getLegalPacmanActions()
        
        #Remove STOP (to make it slighly better)
        if Directions.STOP in legal and len(legal) > 1:
            legal.remove(Directions.STOP)
        
        #Choose random action
        return random.choice(legal)