from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.metrics import recall_score, precision_score
import numpy as np

def get_score(ypred, y):
    """
    Calculating the clustering accuracy. The predicted result must have the same number of clusters as the ground truth.

    ypred: 1-D numpy vector, predicted labels
    y: 1-D numpy vector, ground truth

    The problem of finding the best permutation to calculate the clustering accuracy is a linear assignment problem.
    This function construct a N-by-N cost matrix, then pass it to scipy.optimize.linear_sum_assignment to solve the assignment problem.

    """
    assert len(y) > 0
    assert len(np.unique(ypred)) == len(np.unique(y))

    s = np.unique(ypred)
    t = np.unique(y)

    N = len(np.unique(ypred))
    C = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            idx = np.logical_and(ypred == s[i], y == t[j])
            C[i][j] = np.count_nonzero(idx)

    # convert the C matrix to the 'true' cost
    Cmax = np.amax(C)
    C = Cmax - C
    #
    indices = linear_assignment(C)
    print(indices)
    row = indices[:][:, 0]
    col = indices[:][:, 1]
    # calculating the accuracy according to the optimal assignment
    count = 0
    recall = 0.0
    precision = 0.0
    for i in range(N):
        idx = np.logical_and(ypred == s[row[i]], y == t[col[i]])
        i_correct = np.count_nonzero(idx)
        i_recall_len = np.count_nonzero(y == t[col[i]])
        i_precision_len = np.count_nonzero(ypred == t[col[i]])
        i_precision = float(i_correct) / i_precision_len
        i_recall = float(i_correct) / i_recall_len
        recall += i_recall
        precision += i_precision
    # for i in range(N):
    #     idx = np.logical_and(ypred == s[row[i]], y == t[col[i]])
    #     count += np.count_nonzero(idx)

    return recall/N, precision/N