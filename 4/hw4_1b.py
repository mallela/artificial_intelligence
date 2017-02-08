######################################################
############## Python 2.7#############################
##              CS534 Assignment 4					##
##            Q.No 1(b)								##
##      author: Praneeta Mallela; pmallela@wpi.edu  ##
######################################################

import numpy as np
import itertools
import matplotlib.pyplot as plt
import random
from time import time

# This fn computes the center that the given data is closest to
# this closest cluster is then returned as it's cluster
def getCostFunction(data, k_centers):
	dist_list=[]
	for i in range(len(k_centers)):
		dist = pow(data[0]-k_centers[i][0], 2)+pow(data[1]-k_centers[i][1], 2)
		dist_list.append(dist)
	return dist_list.index(min(dist_list))

# this fn generates k random cenetrs for the given data
# it returns the k centers
def randomCenters(k):
	x=[]
	for i in range(k):
		x.append(tuple((random.uniform(0,4),random.uniform(-1,1))))
	print "The k- random initial centers are: \n", x
	return x

# this fn classifies data into k clusters. It returns the
# new cluster centers and the classified data
def k_means(data,k,no_of_iterations):
	k_centers=randomCenters(k) # generate k random centers
	for n in range(no_of_iterations):
		old_k_centers = k_centers
		for key, value in data.items():
			data[key] = getCostFunction(key,k_centers)
		l = len(k_centers)
		for i in range(l):
			x_average =0; y_average =0;
			same_cluster_data = [k for k,v in data.iteritems() if v==i]
			if same_cluster_data:
				x_average = np.mean([itr[0] for itr in same_cluster_data])
				y_average = np.mean([itr[1] for itr in same_cluster_data])
				k_centers[i] = tuple(((x_average),(y_average)))
	print "The new cluster centers are: \n", k_centers
			# if old_k_centers==k_centers:
			# 	print "breaking...", n
			# 	break
	return k_centers, data


if __name__=='__main__':
	filename = "./realdata.txt"
	data = [float(number)
		for line in open(filename, 'r')
		for number in line.split()]
	data = np.split(np.array(data), len(data)/3,axis=0)
	x = [length[1] for length in data]
	y = [width[2] for width in data]
	data = [tuple((a,b)) for a,b in zip(x,y)]
	# create a dictionary that has {data:cluster it belongs to} for easy reference
	data = {c:0 for c in data }
	k = 7; no_of_iterations = 100
	# if illegal k values are provided, provide a warning and break
	if k ==0 or k<0:
		print "Please provide a positive integer k value"
		exit()
	t0=time()
	k_centers_final, data_clustered = k_means(data, k,no_of_iterations)
	print "Time taken (s):", time()-t0

	# clustering visualization
	colour_list = ['ro', 'bo', 'go','co',  'yo', 'ko','mo']
	colour_list_centers = ['ro', 'bo', 'go', 'co','yo', 'ko','mo']
	for itr in range(k):
		x = [ key[0] for key,value in data_clustered.iteritems() if value==itr ]
		y = [ key[1] for key,value in data_clustered.iteritems() if value==itr ]
		center_x = k_centers_final[itr][0]
		center_y = k_centers_final[itr][1]
		plt.plot(x,y, colour_list[itr], markersize = 4)
		plt.plot(center_x,center_y, colour_list_centers[itr],markersize=10)
		txt_cluster = 'Cluster%s'%(itr)
		plt.text(center_x-.1, center_y+.05,txt_cluster,fontsize=14, weight = 'bold')
	plt.xlabel('Length')
	plt.ylabel('Width')

	plt.show()

