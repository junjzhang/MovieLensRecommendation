import numpy as np
from utils import set_all_random_seed


def load_data_100k(path, seed=42, delimiter='\t'):
    set_all_random_seed(seed)
    train = np.loadtxt(path / 'movielens_100k_u1.base',
                       skiprows=0,
                       delimiter=delimiter).astype('int32')
    test = np.loadtxt(path / 'movielens_100k_u1.test',
                      skiprows=0,
                      delimiter=delimiter).astype('int32')

    n_u = len(set(np.unique(train[:, 0]))
              | set(np.unique(test[:, 0])))  # num of users
    n_m = len(set(np.unique(train[:, 1]))
              | set(np.unique(test[:, 1])))  # num of movies
    n_train = train.shape[0]  # num of training ratings
    n_test = test.shape[0]  # num of test ratings

    val_idx = np.random.choice(n_train, int(n_train * 0.1), replace=False)
    val = train[val_idx]
    n_val = val.shape[0]
    train = np.delete(train, val_idx, axis=0)
    n_train = train.shape[0]

    train_R = np.zeros((n_m, n_u), dtype='float32')
    val_R = np.zeros((n_m, n_u), dtype='float32')
    test_R = np.zeros((n_m, n_u), dtype='float32')

    for i in range(n_train):
        train_R[train[i, 1] - 1, train[i, 0] - 1] = train[i, 2]

    for i in range(n_val):
        val_R[val[i, 1] - 1, val[i, 0] - 1] = val[i, 2]

    for i in range(n_test):
        test_R[test[i, 1] - 1, test[i, 0] - 1] = test[i, 2]

    train_M = np.greater(train_R, 1e-12).astype(
        'float32')  # masks indicating non-zero entries
    test_M = np.greater(test_R, 1e-12).astype('float32')
    val_M = np.greater(val_R, 1e-12).astype('float32')

    print('data matrix loaded')
    print('num of users: {}'.format(n_u))
    print('num of movies: {}'.format(n_m))
    print('num of training ratings: {}'.format(n_train))
    print('num of validation ratings: {}'.format(n_val))
    print('num of test ratings: {}'.format(n_test))

    return n_m, n_u, train_R, train_M, val_R, val_M, test_R, test_M
