from .base import BaseHeuristic
from sokoban.environment import SokobanState

class HungarianHeuristic:
	""" A hungarian heuristic """
	def __call__(self, state: SokobanState, metrics):
	    
	"""
	boxes = state.boxes
	targets = state.targets
	metrics = Manhattan or Euclidean
	"""

	    def getDistances(metrics, box, target):
	        from scipy.spatial import distance
	        if metrics == 'Manhattan':
	            return distance.cityblock(box, target)
	        elif metrics == 'Euclidean':
	            return distance.seuclidean(box, target)



	costMatrix = np.zeros((len(state.boxes), len(state.targets)))

	for i, box in enumerate(state.boxes):
		for j, target in enumerate(state.targets):
			costMatrix[i][j] = getDistances(metrics, box, target)  


	hungarian = Hungarian(cost_matrix) #"http://github.com/tdedecko/hungarian-algorithm"
	hungarian.calculate()

	return float (hungarian.get_total_potential())