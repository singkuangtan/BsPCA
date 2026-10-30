# BsPCA
Use PCA to train deep learning network layer by layer
------------

```python
if i==0: #max_iter-1:
    n_components = 2000 #150 #500 #4000 #200 #500
else:
    n_components = 2000 #150 #500 #4000 
```

Change the above to change the number of components for PCA

```python
def sort_cols_by_l2(matrix,X):    
    # Cosine distance from the first column
    distances = cdist(
      matrix.T,
      matrix[:, 2316].reshape(1, -1), #2316 #999 #99 #9 #1234 #123 #789 #111
      metric='cosine'
    ).flatten()

    # Sort columns by distance
    sorted_matrix = X[:,np.argsort(distances)]
    sorted_matrix2=matrix[:,np.argsort(distances)]
    dist2=np.sort(distances)
    print(dist2[0:20])
    return sorted_matrix,sorted_matrix2
```

Change the number 2316 above to change the index of the query image for nearest matches search

```python
##X_pca=1*(np.random.normal(0,1,size=X_pca.shape)) #0,1 #generate random latent vectors

##X,X_pca=sort_cols_by_l2(X_pca,X)#rank the X_pca vectors to the first vector
```

Uncomment either one above, but not both, to enable random generation of latent vectors to generate random object images, or find top 20 matches based on query image

----------
links to paper https://zenodo.org/records/23733976
