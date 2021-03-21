import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import combinations
from cluster_utils import *

    
if __name__ == "__main__":
    df = pd.read_csv('data\marvel stats\charcters_stats.csv')
    df = df[df.Total != 5].dropna() # Remove default and na rows

    #Labels corresponding to numerical values we want to cluster on
    labels = ['Intelligence','Strength','Speed','Durability','Power','Combat']
    n_centers = 3
    fig, axs = plt.subplots(3, 5)
    min_dist_avgs = {}
    
    #Iterate over the possible pairings for each of the columns
    for (a, b), ax in zip(combinations(labels, 2), axs.flat):
        pts = df[[a,b]].to_numpy()
        centers, labels = gonzalloyds(pts, n_centers)
        plot_clusters(ax, labels, pts)
        key = f'{a} ✕ {b}'
        ax.set_title(key)
        min_dist_avgs[key] = avg_min_cluster_dist(labels, pts)

    for key, val in min_dist_avgs.items():
        print(f'{key} Inter-Cluster Min Distance:\n{val}')

    #Remove tick-labels from upper-left subplots
    for ax in axs.flat:
        ax.label_outer()

    plt.show()
