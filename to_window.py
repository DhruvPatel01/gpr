import numpy as np

def to_window_cbow(matrix, max_window=10):
    M, N  = matrix.shape
    vectors = []
    for i in range(1, max_window+1):
        for j in range(M - i+1):
            vec = matrix[j:j+i, :].sum(0)
            vectors.append(vec)

    return np.array(vectors)