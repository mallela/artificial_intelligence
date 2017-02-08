##################################
# 8 PUZZLE : A* VS RBFS
# Python 2.7.4
# Written by Praneeta Mallela
# last modified: Oct 1 2017
##################################
import heapq
import numpy as np
import random
import copy
import math

def calculateN(state,n):
	N=0
	for i in range(0,n-2):
		current	= state[i]
		for j in range(i+1,n):
			if state[j]<current:
				N+=1
	return N
def checkIfEvenSet(state,n):
	return calculateN(state,n)%2==1

def generateInitialState(n):
	while True :		
		initialState = (random.sample((0,1,2,3,4,5,6,7,8),9))
		if checkIfEvenSet(initialState,n):
			break
	return initialState

def isEmpty(queue):
	return len(queue)==0

def isGoal(state):
	return state==[0,1,2,3,4,5,6,7,8]
def eucledianHeuristic(state):
	return math.sqrt((state[0]-goal[0])**2+(state[1]-goal[1]**2))

def displacementIndicatorHeuristic(state):
	#  find the manhattan distance of each misplaced tile and find sum
	manhattan = 0
	goalState = [(0,1,2),(3,4,5),(6,7,8)]
	state = np.split(np.array(state),3)
	for i in range(0,3):
		for j in range(0,3):
			current = state[i][j]
			actual = np.where(goalState ==current)
			manhattan = manhattan + abs(i-actual[0])+abs(j-actual[1])
	return manhattan
def getSuccessors8Puzzle(state):
	successors = []
	# print "state" ,state
	indx = state.index(0)
	# tempState = state
	if indx ==0: #1,3
		tempState1 = copy.copy(state)
		tempState2 = copy.copy(state)

		tempState1[indx], tempState1[1] = tempState1[1], tempState1[indx]
		tempState2[indx], tempState2[3] = tempState2[3], tempState2[indx]
		
		successors.append((tempState1))
		successors.append((tempState2))
		return successors

	if indx ==1: # 0,2,4
		tempState1 = copy.copy(state)
		tempState2 = copy.copy(state)
		tempState3 = copy.copy(state)

		tempState1[indx], tempState1[0] = tempState1[0], tempState1[indx]
		tempState2[indx], tempState2[2] = tempState2[2], tempState2[indx]
		tempState3[indx], tempState3[4] = tempState3[4], tempState3[indx]

		successors.append((tempState1))
		successors.append((tempState2))
		successors.append((tempState3))

		return successors

	if indx ==2: #1,5
		tempState1 = copy.copy(state)
		tempState2 = copy.copy(state)

		tempState1[indx], tempState1[1] = tempState1[1], tempState1[indx]
		tempState2[indx], tempState2[5] = tempState2[5], tempState2[indx]

		successors.append((tempState1))
		successors.append((tempState2))
		return successors

	if indx ==3: # 0,4,6
		tempState1 = copy.copy(state)
		tempState2 = copy.copy(state)
		tempState3 = copy.copy(state)

		tempState1[indx], tempState1[0] = tempState1[0], tempState1[indx]
		tempState2[indx], tempState2[6] = tempState2[6], tempState2[indx]
		tempState3[indx], tempState3[4] = tempState3[4], tempState3[indx]

		successors.append((tempState1))
		successors.append((tempState2))
		successors.append((tempState3))
		return successors

	if indx ==4: # 1,3,5,7
		tempState1 = copy.copy(state)
		tempState2 = copy.copy(state)
		tempState3 = copy.copy(state)
		tempState4 = copy.copy(state)

		tempState1[indx], tempState1[1] = tempState1[1], tempState1[indx]
		tempState2[indx], tempState2[3] = tempState2[3], tempState2[indx]
		tempState3[indx], tempState3[5] = tempState3[5], tempState3[indx]
		tempState4[indx], tempState4[7] = tempState4[7], tempState4[indx]

		successors.append((tempState1))
		successors.append((tempState2))
		successors.append((tempState3))
		successors.append((tempState4))
		return successors

	if indx ==5: # 2,4,8
		tempState1 = copy.copy(state)
		tempState2 = copy.copy(state)
		tempState3 = copy.copy(state)

		tempState1[indx], tempState1[2] = tempState1[2], tempState1[indx]
		tempState2[indx], tempState2[8] = tempState2[8], tempState2[indx]
		tempState3[indx], tempState3[4] = tempState3[4], tempState3[indx]

		successors.append((tempState1))
		successors.append((tempState2))
		successors.append((tempState3))
		return successors
	if indx == 6: # 3,7
		tempState1 = copy.copy(state)
		tempState2 = copy.copy(state)
		
		tempState1[indx], tempState1[3] = tempState1[3], tempState1[indx]
		tempState2[indx], tempState2[7] = tempState2[7], tempState2[indx]
		
		successors.append((tempState1))
		successors.append((tempState2))
		return successors
	if indx == 7: # 4,6,8
		tempState1 = copy.copy(state)
		tempState2 = copy.copy(state)
		tempState3 = copy.copy(state)
		
		tempState1[indx], tempState1[6] = tempState1[6], tempState1[indx]
		tempState2[indx], tempState2[8] = tempState2[8], tempState2[indx]
		tempState3[indx], tempState3[4] = tempState3[4], tempState3[indx]
		
		successors.append((tempState1))
		successors.append((tempState2))
		successors.append((tempState3))
		return successors
	if indx == 8: # 5,7
		tempState1 = copy.copy(state)
		tempState2 = copy.copy(state)

		tempState1[indx], tempState1[5] = tempState1[5], tempState1[indx]
		tempState2[indx], tempState2[7] = tempState2[7], tempState2[indx]
		
		successors.append((tempState1))
		successors.append((tempState2))
		return successors

	# return successors
def getSuccessorsTSP(state):
	return 1
def astar(startState, initialCost):
	closedSet = {}
	totalNodesExpanded = 0
	closedSet[tuple(startState)]=initialCost
	fringe = []
	heapq.heappush(fringe, (initialCost,startState,startState))
	while (not isEmpty(fringe)):
		node = heapq.heappop(fringe)
		print node
		totalNodesExpanded+=1
		if isGoal(node[1]):
			print "Total nodes expanded are",totalNodesExpanded
			print "Solution is \n",np.split(np.array(node[2]), np.size(node[2])/9)
			print "Success! Goal reached in ", np.size(node[2])/9," steps!"
			return
		for successor in getSuccessors8Puzzle(node[1]):
			print
			g = node[0]+1
			h = displacementIndicatorHeuristic(node[1])
			f = g+h
			if tuple(successor) in closedSet:
				if f<closedSet[tuple(successor)]: 
					closedSet[tuple(successor)]=f
					heapq.heappush(fringe,(f,successor,node[2]+[successor]))
			else:
				if isGoal(successor):
					print "Total nodes expanded are",totalNodesExpanded
					print "Solution is \n",np.split(np.array(node[2]+successor), np.size(node[2]+successor)/9)
					print "Success! Goal reached in ", np.size(node[2]+successor)/9," steps!"
					return
				closedSet[tuple(successor)]=f
				heapq.heappush(fringe,(f,successor,node[2]+[successor]))
	return 1

def rbfsSearch(startState):
	# flimit = np.inf
	fringe=[]

	def rbfs(state,flimit):
		fringe = []
		node = state

		if isGoal(node[1]): 
			return node[1] 
		successors = []
		# print "node1", node[1]
		successors = getSuccessors8Puzzle(node[1])
		if not len(successors): 
			print "empty successors"
			return None, np.inf
		print "___"
		for successor in successors:
			print successor
			successor_g = node[0]-displacementIndicatorHeuristic(node[1])+1
			successor_h = displacementIndicatorHeuristic(successor)
			successor_f = max(successor_g+successor_h,node[0])
			# print 
			heapq.heappush(fringe,(successor_f,successor))
		while True:
			print len(fringe)
			# best = heapq.heappop(fringe)
			if isEmpty(fringe): return None, np.inf
			best = fringe[0]

			if best[0]> flimit:
				# print True
				return None, best[0]
			# if isEmpty(fringe): return None, np.inf
			print "jhad", fringe
			if len(fringe)>1: alternative = fringe[1]
			if len(fringe)==1: alternative = fringe[0]
			print "here",np.array(best)[0], best, alternative[0]
			heapq.heappop(fringe)
			# print fringe

			result, np.array(best)[0] = rbfs(best, min(flimit, alternative[0]))
			if result is not None: return result, "sA"
	cost = displacementIndicatorHeuristic(startState)
	return rbfs((cost,startState),np.inf)
if __name__=='__main__':
	x= generateInitialState(9)
	infinity = np.inf
	print "initial state is \n ",x,"\n_____________________________________"
	print "ASTAR...."
	astar(x, calculateN(x,9))
	print "_____________________________________"
	# print "RBFS...."
	# print rbfsSearch(x)
