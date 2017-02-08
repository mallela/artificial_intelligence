##################################
# HILL-CLIMBING SEARCH ALGO for TSP
# Python 2.7.4
# Written by Praneeta Mallela
# last modified: Oct 1 2017
##################################
import numpy as np
import math
import copy, operator, itertools, random
mapOfTSP = {1:(1,0),2:(0,1),3:(2,1),4:(1,2),5:(1,3),
6:(2,3),7:(3,3),8:(4,1),9:(4,2)}
goal = mapOfTSP[9] 

connections =[(0, 1, np.inf, 1, 1, np.inf, np.inf, np.inf, np.inf),
(1,0,1,np.inf,np.inf,np.inf,np.inf,1,np.inf),
(np.inf,1,0,1,np.inf,np.inf,1,np.inf,np.inf),
(1,np.inf,1,0,np.inf,np.inf,np.inf,np.inf,np.inf),
(1,np.inf,np.inf,np.inf,0,1,np.inf,1, np.inf),
(np.inf,np.inf,np.inf,np.inf,1,0,1,np.inf,np.inf),
(np.inf,np.inf,1,np.inf,np.inf,1,0,np.inf,1),
(np.inf,1,np.inf,np.inf,1,np.inf,np.inf,0,1),
(np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,1,1,0)]

def eucledianDist(tour):
	temp = tour
	pathCost = 0
	for i in range(0,len(tour)-2):
		x = temp[i]; y = temp[i+1]
		if connections[x-1][y-1]==np.inf:
			return np.inf
		else:
			xy1 = mapOfTSP[x]; xy2 = mapOfTSP[y]
			pathCost +=math.sqrt((xy1[0]-xy2[0])**2+(xy1[1]-xy2[1])**2)#**0.5
	x = temp[0]; y = temp[len(tour)-1]
	return pathCost+ math.sqrt((xy1[0]-xy2[0])**2+(xy1[1]-xy2[1])**2)


def hillClimbing(startState):
	tourDistance = {}
	currentTour = startState
	for i in range(1,len(currentTour)-2):
		tourCombinations = getSuccessorsTSP(currentTour, i)
		for alternateTour in tourCombinations:
			tourDistance[tuple(alternateTour)] = eucledianDist(alternateTour)
		numberOfLoops = len(tourCombinations)
		while numberOfLoops>=1: 
			highestvaluePath = min(tourDistance.iteritems(), key=operator.itemgetter(1))[0]
			highestvaluePathCost = tourDistance[highestvaluePath] 

			if highestvaluePathCost<= eucledianDist(currentTour):
				currentTourTemp = currentTour
				currentTour = highestvaluePath
			else:
				return currentTour, eucledianDist(currentTour)
			numberOfLoops-=600
		print currentTour, eucledianDist(currentTour)
	return currentTour,eucledianDist(currentTour)
		
def getSuccessorsTSP(tour,city):
	alternateTour = []
	tourTemp = copy.copy(tour)
	permutationsOfTours = [ x for x in list(itertools.permutations(tourTemp)) if (x[0]==x[len(x)-1]==1 and x[1]==city+1)]
	return permutationsOfTours

if __name__=='__main__':
	tour,cost =  hillClimbing([1,2,3,4,5,6,7,8,9,1]) 
	print "The solution to the TSP using hill-climbing is",tour,"with a cost of", cost
