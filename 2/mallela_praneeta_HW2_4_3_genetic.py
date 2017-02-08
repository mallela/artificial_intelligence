##################################
# GENETIC SEARCH for TSP
# Python 2.7.4
# Written by Praneeta Mallela
# last modified: Oct 1 2017
##################################
import numpy as np
import math, copy
import random, itertools

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

def getSuccessorsTSP(tour,city,tf):

	alternateTour = []
	tourTemp = copy.copy(tour)
	if tf==0:
		permutationsOfTours = [ x for x in list(itertools.permutations(tourTemp)) if (x[0]==x[len(x)-1]==1 and x[1]==city+1)]
	else:
		permutationsOfTours = [x for x in list(itertools.permutations(tourTemp)) if (x[0]==x[len(x)-1]==1 and eucledianDist(x)!=np.inf)]
	return permutationsOfTours

def eucledianDist(tour):
	temp = tour;pathCost = 0
	for i in range(0,len(tour)-2):
		x = temp[i]; y = temp[i+1]
		if connections[x-1][y-1]==np.inf:
			return np.inf
		else:
			xy1 = mapOfTSP[x]; xy2 = mapOfTSP[y]
			pathCost +=math.sqrt((xy1[0]-xy2[0])**2+(xy1[1]-xy2[1])**2)
	x = temp[0]; y = temp[len(tour)-1]
	return pathCost+ math.sqrt((xy1[0]-xy2[0])**2+(xy1[1]-xy2[1])**2)

def randomSelection(tours, fitnessFunc,n):
	total = 0; fitness=[]; selections = []
	for i in range(len(tours)):
		fitness.append(fitnessFunc(tours[i]))
		total +=fitnessFunc(tours[i])
	while not len(selections):
		for s in range(n):
			r = random.uniform(0,fitness[-1])
			for i in range(len(tours)):
				if fitness[s]>=r:
					selections.append(tours[s])
					break
	return selections

def reproduce(p1, p2):
	c = random.randint(0,len(p1))
	return p1[:c]+p2[c:]

def mutate(childTour):
	randIndex = random.randint(1,len(childTour)-2)
	randNum = random.randint(1,len(childTour)-1)
	return [randNum if x == childTour[randIndex] else x for x in childTour]

def bestIndividual(population, fitnessFunc):
	fitness = [];total = 0
	for i in range(len(population)):
		fitness.append(fitnessFunc(population[i]))
		total +=fitnessFunc(population[i])
	indx = fitness.index(max(fitness))
	return population[indx]

def geneticAlgorithm(population,fitnessFunc,time):
	for i in range (1,time):
		print "Iteration number", i, "out of", time
		newPopulation = []
		for i in range(len(population)):
			parents = randomSelection(population, fitnessFunc,2)
			child = reproduce(parents[0],parents[1])
			if random.uniform(0,1)>0.4:
				child = mutate(child)
			newPopulation.append(child)
		del population[population.index(parents[1])]
		population+=newPopulation
	return bestIndividual(population, fitnessFunc)

if __name__== '__main__':
	sizeOfPoplation = 20;pmut = 0.4
	tours = [];population = [];initialTour = [1,2,3,4,5,6,7,8,9,1];populationPool=[]
	populationPoolFinite = [x for x in getSuccessorsTSP(initialTour,0, 1) ]
	populationPoolInfinite = random.sample([x for x in getSuccessorsTSP(initialTour,random.randint(1,len(initialTour)-2), 0)],sizeOfPoplation-len(populationPoolFinite))
	populationPool = populationPoolFinite+populationPoolInfinite
	random.shuffle(populationPool)
	fitnessFunc = (lambda s : (- eucledianDist(s)))
	bestIndi = geneticAlgorithm(populationPool,fitnessFunc, time=10)
	print "The solution to the TSP problem using Genetic Search algo is ",bestIndi,"with a cost of", (-fitnessFunc(bestIndi))
