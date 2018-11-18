#initial value
missing = ~np.isfinite(X)
mu = np.nanmean(X, 0, keepdims=1)
X_hat = np.where(missing, mu, X).reshape(-1, 1)

for i in xrange(10):
    if i > 0:
        cls = KMeans(n_clusters, init=prev_centroids)
    else:
        # do multiple random initializations in parallel
        cls = KMeans(n_clusters, n_jobs=-1)

    # perform clustering on the filled-in data
    labels = cls.fit_predict(X_hat)
    centroids = cls.cluster_centers_

    # fill in the missing values based on their cluster centroids
    X_hat[missing] = centroids[labels][missing]

    # when the labels have stopped changing then we have converged
    if i > 0 and np.all(labels == prev_labels):
        break

    prev_labels = labels
    prev_centroids = cls.cluster_centers_
