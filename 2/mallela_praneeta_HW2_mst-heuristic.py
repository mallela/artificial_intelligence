#######################################
# A* for TSP using MST by Kruskal 
# Python 2.7.4
# Written by Praneeta Mallela
# last modified: Oct 1 2017
#######################################
import heapq, math, operator, copy
import numpy as np
mapOfTSP = {1:(1,8),2:(5,9),3:(2,1),4:(1,2),5:(1,3),
6:(2,3),7:(3,3),8:(4,1),9:(4,2)}

connections =[(0, 1, np.inf, 1, 1, np.inf, np.inf, np.inf, np.inf),
(1,0,1,np.inf,np.inf,np.inf,np.inf,1,np.inf),
(np.inf,1,0,1,np.inf,np.inf,1,np.inf,np.inf),
(1,np.inf,1,0,np.inf,np.inf,np.inf,np.inf,np.inf),
(1,np.inf,np.inf,np.inf,0,1,np.inf,1, np.inf),
(np.inf,np.inf,np.inf,np.inf,1,0,1,np.inf,np.inf),
(np.inf,np.inf,1,np.inf,np.inf,1,0,np.inf,1),
(np.inf,1,np.inf,np.inf,1,np.inf,np.inf,0,1),
(np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,1,1,0)]

def getSuccessors(partialPath):
	try:
		return [x for x in mapOfTSP if x not in partialPath ]
	except TypeError:
		l = []; l.append(partialPath); print "l",l
		return [x for x in mapOfTSP if x not in l ]		

def isEmpty(queue):
	return len(queue)==0
def eucledianDist(twoCities):
	city1,city2 = twoCities
	xy1 = mapOfTSP[city1]; xy2 = mapOfTSP[city2]

	return math.sqrt((xy1[0]-xy2[0])**2+(xy1[1]+xy2[1])**2)
def isGoal(state):
	s = []; s.append(state)
	return len(s)== len(mapOfTSP)

def heuristic(partialPath):
	mst = 0
	toBeVisited = [x for x in mapOfTSP if x not in np.array(partialPath)]
	edgeSetWithCost = {}
	print partialPath
	edgeFunc = (lambda s : eucledianDist(s)) 
	if len(toBeVisited)>1:
		for i in range(0,len(toBeVisited)-1):
			for j in range(i+1, len(toBeVisited)):
				edgeSetWithCost[toBeVisited[i],toBeVisited[j]]=eucledianDist((toBeVisited[i],toBeVisited[j]))
	elif len(toBeVisited)==1:
		edgeSetWithCost[toBeVisited[0],partialPath[-1]]=eucledianDist((toBeVisited[0],partialPath[-1]))
	else:
		edgeSetWithCost[1,partialPath[-1]]=eucledianDist((1,partialPath[-1]))
	mst += min(edgeSetWithCost.iteritems(), key=operator.itemgetter(1))[1]
	return mst

def astar (startState, initialCost):
	closedSet = {}
	# print startState
	totalNodesExpanded = 0
	closedSet[tuple(startState)]=initialCost
	fringe = []
	heapq.heappush(fringe, (initialCost,startState,startState))
	while (not isEmpty(fringe)):
		node = heapq.heappop(fringe)
		# print node

		totalNodesExpanded +=1
		if isGoal(node[2]): 
			solution = node[2]
			print "Total nodes expanded are",totalNodesExpanded
			print "Solution is \n",np.split(np.array(solution), np.size(solution)/6)
			print "Success! Goal reached in ", np.size(solution)/6," steps!"
			return
		for successor in getSuccessors(node[2]):
			# print "nde", node[2]
			print "cost",node[0]
			g = node[0] +connections[node[1][0]-1][successor-1]
			h = heuristic(node[2]+[successor])
			f = g + h
			if f!=np.inf:
				s= node[2]+[successor]
				if tuple(s) in closedSet:
					if closedSet[s]> f: 
						closedSet[s] = f
						heapq.heappush(fringe, (f,[successor], s))
				else:
					if isGoal(node[2]+[successor]): # checking for goal state here reduces the number of nodes expanded
						print "Total nodes expanded are",totalNodesExpanded +1 # adding one to count generated state the goal state too
						solution= node[2]+[successor]
						print "Solution is \n",np.split(np.array(solution), np.size(solution)/6)
						print "Success! Goal reached in ", np.size(solution)/6," steps!"
						return
					closedSet[tuple(s)] = f
					heapq.heappush(fringe, (f,[successor],s))
					
	print "Total nodes expanded are",totalNodesExpanded +1 # adding one to count generated state the goal state too
	return node[2]+[1]
if __name__ == '__main__':
	startState = [1]
	initialCost = 0+heuristic(copy.copy(startState)) 
	print astar(startState,initialCost)
	