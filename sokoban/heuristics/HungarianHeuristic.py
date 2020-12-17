from .base import BaseHeuristic
from sokoban.environment import SokobanState
from sokoban.heuristics.hungarian import Hungarian
from fastdist import fastdist
import numpy as np

def getDistances(metrics, box, target):
	from scipy.spatial import distance
	if metrics == 'Manhattan':
		return distance.cityblock(box, target)
	elif metrics == 'Euclidean':
		return fastdist.euclidean(box, target)

class HungarianHeuristic(BaseHeuristic):
	""" A hungarian heuristic """
	def __init__(self, metrics):
		super(HungarianHeuristic, self).__init__()

		self.metrics = metrics

	def __call__(self, state: SokobanState): 
		"""
		boxes = state.boxes
		targets = state.targets
		metrics = Manhattan or Euclidean
		"""
		costMatrix = np.zeros((len(state.boxes), len(state.targets)))

		for i, box in enumerate(state.boxes):
			for j, target in enumerate(state.targets):
				costMatrix[i][j] = getDistances(self.metrics, box, target)  


		hungarian = Hungarian(costMatrix) #"http://github.com/tdedecko/hungarian-algorithm"
		hungarian.calculate()

		return float (hungarian.get_total_potential())