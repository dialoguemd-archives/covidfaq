# Covid19

Task - Sorting Questions into Groups/Clusters such that the whole group can be asnwered with a single response.

## Short and Sweet
The files you are most interested in, that contain the latest experiemnts are the `Pipeline` notebooks. 

## Methodology
The way I analyse clusters - 
1. Cluster using LSA + AHC with low enough distance threshold to get clusters such that the cluster distribution is close to the following stats - 
    ```
    count    43.000000
    mean     11.069767
    std       8.209737
    min       2.000000
    25%       5.000000
    50%       8.000000
    75%      14.500000
    ```
The reason for this is to get closely knit clusters to see which questions are clustered together and probably why. "Why" can help determine the rules for the next step.  
2. Create rules that can separate out the major clusters from the step #1. Check the rules.  
3. Repeat the step #1 and #2 on the unclustered to observe the patterns, while lowering the distance threshold.

### Folders
1. `data` folder contains the files that are used as input for data analysis
1. `collab_data` folder contains other files that were passed on in the Covid-19_task_force slack
1. `output` folder contains the resultant csvs
1. `utils` folder contains subset of personal scripts out of which I used scripts for Doc2Vec and LSI

### Files
The purpose of different python notebooks is - 
1. `Doc2Vec` notebook contains the first experiments conducted for clustering. Appraoch used is Doc2Vec with Kmeans and Agglomerative clustering.
1. `Topic Modelling` notebook contains experiments using topic distribution representation with different clustering methods. LSI and LDA are tried for generating topic distributions, and KMeans and Agglomerative clustering are tried on top of topic distribution representation.
