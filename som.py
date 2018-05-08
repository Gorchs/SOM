#importing the libraries
import matplotlib.pyplot as plt
import matplotlib.colors as colormap
import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D


def create_dataset(n):
    p=[]
    for i in range(n):
      Pi=[random.randint(1,1001),random.randint(1,1001),random.randint(1,1001)]
      p.append(Pi)
    return(p)
x=create_dataset(500)

#Finding the optimum number of clusters for k-means classification
from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    y=kmeans.fit(x)
    wcss.append(kmeans.inertia_)
'''
#Plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') #within cluster sum of squares
plt.grid(color='r', linestyle='-', linewidth=1) #within cluster sum of squares
plt.figure(figsize=(6,8))
plt.show()
'''
kmeans = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
kmeans = kmeans.fit(x)
labels = kmeans.predict(x)

centroids = kmeans.cluster_centers_
labels=kmeans.labels_


#x=np.asarray(x)
fig = plt.figure()
ax = Axes3D(fig)


X=[];
Y=[];
Z=[];
colors=[];
for i in range(0, len(x)):
    X.append(x[i][0]);
    Y.append(x[i][1]);
    Z.append(x[i][2]);
    if labels[i]==0:
        colors.append('r');
    if labels[i]==1:
        colors.append('g');



for i in range(len(x)):
    ax.scatter(X[i], Y[i], Z[i], s=3, color=colors[i])

#Plotting the centroids of the clusters
ax.scatter(centroids[:, 0], centroids[:,1],centroids[:,2], s = 30, c = ['r','g'], label = 'Centroids')
ax.legend()
plt.show()

