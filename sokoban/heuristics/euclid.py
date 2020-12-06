from .base import BaseHeuristic
from sokoban.environment import SokobanState
import numpy as np
import math

class EuclidHeuristic:
    """ A euclid greedy heuristic. """
    def __call__(self, state: SokobanState) -> float:
        result = 0.0
        box_target = []
        for i in state.boxes:
            box_target.append (-1)
                    
        for i in range (len (state.boxes)):
            box = state.boxes[i]
            if box_target[i] >= 0: continue
            distances = []
            for target in state.targets:
                dist = math.sqrt ( (target[0]-box[0])**2 + (target[1]-box[1])**2 )
                distances.append (float (dist))
            dist = min (distances)
            index = np.argmin (distances)
            while box_target[index] > 0 and len (distances) > 1:
                distances.remove (dist)
                dist = min (distances)
                index = np.argmin (distances)
            
            box_target[index] = dist
                
        return float (sum (box_target) )
