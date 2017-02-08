#######################################
# Missionaries and cannibals using A* 
# Python 2.7.4
# Written by Praneeta Mallela
# last modified: Oct 1 2017
#######################################
import heapq
import numpy as np
def getSuccessors(state):
	# return 1
	if state == (3,3,1,0,0,0) : successors = ((3,1,0,0,2,1),(2,2,0,1,1,1),(3,2,0,0,1,1))
	if state == (3,1,0,0,2,1) or state == (2,2,0,1,1,1) : successors = ((3,2,1,0,1,0),(3,3,1,0,0,0))
	if state == (3,2,0,0,1,1): successors = ((3,3,1,0,0,0))
	if state == (3,2,1,0,1,0): successors = ((3,0,0,0,3,1),(2,2,0,1,1,1))
	if state == (3,0,0,0,3,1): successors = ((3,1,1,0,2,0),(3,2,1,0,1,0))
	if state == (3,1,1,0,2,0): successors = ((1,1,0,2,2,1),(3,0,0,0,3,1))
	if state == (1,1,0,2,2,1): successors = ((2,2,1,1,1,0),(3,1,1,0,2,0))
	if state == (2,2,1,1,1,0): successors = ((0,2,0,3,1,1),(1,1,0,2,2,1))
	if state == (0,2,0,3,1,1): successors = ((0,3,1,3,0,0),(2,2,1,1,1,0))
	if state == (0,3,1,3,0,0): successors = ((0,1,0,3,2,1),(0,2,0,3,1,1))
	if state == (0,1,0,3,2,1): successors = ((0,2,1,3,1,0),(1,1,1,2,2,0),(0,3,1,3,0,0))
	if state == (0,2,1,3,1,0) or state == (1,1,1,2,2,0): successors = ((0,0,0,3,3,1),(0,1,0,3,2,1))
	if state == (0,1,1,3,2,0): successors = (0,0,0,3,3,1)
	return successors
def isEmpty(queue):
	return len(queue)==0

def isGoal(state):
	if state ==(0,0,0,3,3,1):
		return True
def heuristic(state):
	missionaries, cannibals, _,_,_,_ = state
	return missionaries+cannibals

def peopleOnOthersideWithBoat(successor):
	_,_,_,missionaries,cannibals,boat = successor
	return (missionaries+cannibals == 1) and (boat == 1)

def astar (startState, initialCost):
	closedSet = {}
	totalNodesExpanded = 0
	closedSet[startState]=initialCost
	fringe = []
	heapq.heappush(fringe, (initialCost,startState,startState))
	while (not isEmpty(fringe)):
		node = heapq.heappop(fringe)
		totalNodesExpanded +=1
		if isGoal(node[1]): 
			solution = node[2]
			print "Total nodes expanded are",totalNodesExpanded
			print "Solution is \n",np.split(np.array(solution), np.size(solution)/6)
			print "Success! Goal reached in ", np.size(solution)/6," steps!"
			return
		for successor in getSuccessors(node[1]):
			g = node[0]+1 # +1 for uniform unit path cost from 1 node to next
			h = heuristic(successor)
			f = g + h
			if not peopleOnOthersideWithBoat(successor):
				if successor in closedSet:
					if closedSet[successor]> f: 
						closedSet[successor] = f

						heapq.heappush(fringe, (f,successor, node[2]+successor))
				else:
					if isGoal(successor): # checking for goal state here reduces the number of nodes expanded
						print "Total nodes expanded are",totalNodesExpanded +1 # adding one to count expanding the goal state too
						solution= node[2]+successor
						print "Solution is \n",np.split(np.array(solution), np.size(solution)/6)
						print "Success! Goal reached in ", np.size(solution)/6," steps!"
						return
					closedSet[successor] = f
					heapq.heappush(fringe, (f,successor,node[2]+successor))
if __name__ == '__main__':
	startState = (3,3,1,0,0,0)
	initialCost = 0+6 
	astar(startState,initialCost)
	