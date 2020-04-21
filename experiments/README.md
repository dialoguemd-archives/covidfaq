# Covid19 Question Clustering - Experiments

Task - Sorting Questions into Groups/Clusters such that the whole group can be answered with a single response.

## Short and Sweet
The files you are most interested in, that contain the latest experiments are the `Pipeline` notebooks. 

## Methodology
The way I analyze clusters - 
1. Cluster using LSA + AHC. At this step the `distance_threshold` used for agglomerative clustering is selected so that
    1. The clusters are homogeneous - There are fewer false positives in the clusters.
    1. The clusters aren't too small. Example - 1 data point each for 25 clusters, is too small a cluster.
    1. The clusters aren't too many. Example - 300 clusters for 700 datapoint does not make much sense for broad categorization.
1. Create rules that can separate out the major clusters from the step #1. Check the rules.  
1. Repeat the step #1 and #2 on the `unclustered` to observe the patterns, while lowering the `distance threshold`.

### Folders
1. `data` folder contains the files that are used as input for data analysis
1. `collab_data` folder contains other files that were passed on in the Covid-19_task_force slack
1. `output` folder contains the resultant CSVs
1. `utils` folder contains subset of personal scripts out of which I used scripts for Doc2Vec and LSI

### Files
The purpose of different python notebooks is - 
1. `Doc2Vec` notebook contains the first experiments conducted for clustering. Approach used is Doc2Vec with Kmeans and Agglomerative clustering.
1. `Topic Modelling` notebook contains experiments using topic distribution representation with different clustering methods. LSI and LDA are tried for generating topic distributions, and KMeans and Agglomerative clustering are tried on top of topic distribution representation.
1. `LSA and AHC` notebook contains experiments using specifically LSA and AHC tried on 2 levels, one after another.
1. `Pipeline` notebooks contain the experiments in a pipeline of rules followed by LSA and AHC for the data points that did not fall under the rules to look for clusters.
