# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 09:40:28 2019

@author: Administrator
"""

from sklearn.datasets.samples_generator import make_blobs
Data, y=make_blobs(n_samples=600,n_features=2,centers=4)
import matplotlib.pyplot as plt
plt.scatter(Data[:,0],Data[:,1])

#Methode de coude
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

X = Data
distorsions = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    distorsions.append(kmeans.inertia_)
k=range(2,10)
fig = plt.figure(figsize=(15, 5))
plt.plot(k, distorsions)
plt.grid(True)
plt.title('Elbow curve')

#methode de silouette
from sklearn.metrics import silhouette_samples, silhouette_score
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k)
    y_pred=kmeans.fit_predict(Data)
    score=silhouette_score(Data,y_pred)
    print ("For k = {}, silhouette score is {})".format(k, score))
    
from yellowbrick.cluster import SilhouetteVisualizer
# Instantiate the clustering model and visualizer
for k in range(2,10):
    model = KMeans(k, random_state=42)
    plt.subplot(221)
    visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
    visualizer.fit(Data)        # Fit the data to the visualizer
    visualizer.show()        # Finalize and render the figure   
    plt.subplot(222)
    plt.scatter(Data[:,0],Data[:,1],c=y_pred,cmap='rainbow')
    
    
