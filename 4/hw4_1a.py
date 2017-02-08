######################################################
############## Python 2.7#############################
##              CS534 Assignment 4					##
##            Q.No 1(a)								##
##      author: Praneeta Mallela; pmallela@wpi.edu  ##
######################################################

#  k-means if the simplest unsupervised learning algos
import numpy as np
import random 
import itertools
import matplotlib.pyplot as plt

# generates 'k' random centers
# input is k
# returns k random, 3-channel colors
def randomCenters(k):
	x=[]
	for i in range(k):
		x.append(tuple((random.randint(0,255),random.randint(0,255),random.randint(0,255))))
	return x

# this fn classifies the given data into clusters
# until the max iterations is exceeded
# it then returns the new centers and the data in each cluster
def k_means(data,k,no_of_iterations):
	k_centers=randomCenters(k) # generate k random centers
	print "The random k- cluster centers are: "
	for n in range(no_of_iterations):
		for key, value in data.items():
			data[key] = getCostFunction(key,k_centers)
		l = len(k_centers)
		for i in range(l):
			r_average =0; g_average =0; b_average =0;
			same_cluster_data = [k for k,v in data.iteritems() if v==i]
			if same_cluster_data:
				r_average = np.mean([itr[0] for itr in same_cluster_data])
				g_average = np.mean([itr[1] for itr in same_cluster_data])
				b_average = np.mean([itr[2] for itr in same_cluster_data])
				k_centers[i] = tuple((int(r_average),int(g_average), int(b_average)))
		print "The new cluster centers are: ", k_centers
	return k_centers, data

# this fn finds which center the given data is closest to
# it returns the nearest cluster to the data
def getCostFunction(data, k_centers):
	dist_list=[]
	for i in range(len(k_centers)):
		dist = pow(data[0]-k_centers[i][0], 2)+pow(data[1]-k_centers[i][1], 2)+pow(data[2]-k_centers[i][2], 2)
		dist_list.append(dist)
	return dist_list.index(min(dist_list))

# this fn generates a given no. of cololors(data)
# that needs to be classified into k clusters
def getData(data_size):
	data = []
	for i in range(data_size):
		data.append(tuple((random.randint(0,255),random.randint(0,255),random.randint(0,255))))
	return data

if __name__=='__main__':
	k=3 # number of clusters
	data_size = 200 # number of data points
	no_of_iterations = 10 # number of iteratoins for k-means
	data_list = getData(data_size) # generate data in list format
	# format data into dictionary, key = data, value = cluster number
	data = {data_list:0 for data_list in data_list} 
	# do k means to get true centers, and cluster labels for data
	k_centers, data = k_means(data, k,no_of_iterations) 

	# visualization of data after kmeans 
	colours = [];
	beforeKmeans = np.zeros((data_size, 80, 3), dtype=np.uint8) 
	finalClusterCenters = np.zeros((data_size , 80, 3), dtype=np.uint8)
	
	for itr in range(k):
		colors_same_cluster = []
		colors_same_cluster.append([key for key,value in data.iteritems() if value ==itr])
		colors_same_cluster = np.hstack(colors_same_cluster)
		afterKmeans = np.zeros((data_size , 80, 3), dtype=np.uint8)
		
		for x in range(len(colors_same_cluster)):
			afterKmeans[x][:]=[colors_same_cluster[x][0],colors_same_cluster[x][1],colors_same_cluster[x][2]]
		
		plt.figure(itr+1)
		afterKmeans = afterKmeans
		plt.imshow(afterKmeans)
		txt = 'Cluster number %s'%(itr+1)
		plt.title(txt)

	colours.append([key for key,value in data.iteritems()])
	colours_image = np.hstack(colours)
	for x in range(data_size):
		beforeKmeans[x][:]= [colours_image[x][0],colours_image[x][1],colours_image[x][2]]
	for m in range(0,67):
		finalClusterCenters[m][:]= [k_centers[0][0],k_centers[0][1],k_centers[0][2]]
	for n in range(67,133):
		finalClusterCenters[n][:]= [k_centers[1][0],k_centers[1][1],k_centers[1][2]]
	for l in range(133,200):
		finalClusterCenters[l][:]= [k_centers[2][0],k_centers[2][1],k_centers[2][2]]

	colours_image = beforeKmeans
	finalClusterCenters =finalClusterCenters

	plt.figure(itr+2)
	plt.title("Cluster center colours")
	plt.imshow(finalClusterCenters)
	plt.figure(itr+3)
	plt.imshow(colours_image)
	plt.title('Unclustered data')
	plt.show()
