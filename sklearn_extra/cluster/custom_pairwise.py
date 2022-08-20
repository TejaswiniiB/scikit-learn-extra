def pairwise_distances(X, Y, metric):
    import itertools
    # Only calculate metric for upper triangle
    out = np.zeros((X.shape[0], Y.shape[0]), dtype="float")
    iterator = itertools.combinations(range(X.shape[0]), 2)
    for i, j in iterator:
        out[i, j] = metric_callable(X[i], Y[j])

    # Make symmetric
    # NB: out += out.T will produce incorrect results
    out = out + out.T

    # Calculate diagonal
    # NB: nonzero diagonals are allowed for both metrics and kernels
    for i in range(X.shape[0]):
        x = X[i]
        out[i, i] = metric(x, x)

    return out
