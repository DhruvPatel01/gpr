import numpy as np

def to_window_cbow(x, y_ind, y_len, ignore, max_window=10):
    M, N  = x.shape
    new_x = []
    new_y = []
    additional = []
    for i in range(1, max_window+1):
        for j in range(M - i+1):
            if j <= ignore < j+i:
                continue
            if i == y_len and j == y_ind:
                new_y.append(1)
            else:
                new_y.append(0)
            vec = x[j:j+i, :].sum(0)
            new_x.append(vec)
            additional.append((j, i))
    return np.array(new_x), np.array(new_y), additional