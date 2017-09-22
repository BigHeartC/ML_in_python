import numpy as np


class NB_classifier(object):
    '''
    离散特征
    '''

    def __init__(self, lam=1):
        self.lam = lam

    def fit(self, X, y):
        self.train_X = np.array(X)
        self.train_y = np.array(y)

    def predict(self, X):
        X = np.array(X)
        fea_num = self.train_X.shape[1]
        y_uni_len = len(np.unique(self.train_y))
        print('fea_num', fea_num)
        y_pred = np.array([])
        for x_samp in X:
            max_y, max_y_proba = -1, -1
            for y_smap in np.unique(self.train_y):
                tmp_x = self.train_X[self.train_y == y_smap]
                tmp_proba = (len(tmp_x) + 1) * 1.0 / (len(self.train_y) + y_uni_len)
                y_smap_proba = tmp_proba
                print('p( y =', y_smap, ')=', tmp_proba)
                for x_index in range(fea_num):
                    tmp_proba = (sum(tmp_x[:, x_index] == x_samp[x_index]) + 1) / (tmp_x.shape[0] + y_uni_len)
                    print('p( x =', x_samp[x_index], '|y =', y_smap, ')=', tmp_proba)
                    y_smap_proba *= tmp_proba
                print(y_smap, y_smap_proba)
                if max_y_proba < y_smap_proba:
                    max_y, max_y_proba = y_smap, y_smap_proba
            print(max_y)
            y_pred = np.append(y_pred,[max_y])
        return y_pred


if __name__ == '__main__':
    X = np.array([[1, 'S', -1],
                  [1, 'M', -1],
                  [1, 'M', 1],
                  [1, 'S', 1],
                  [1, 'S', -1],
                  [2, 'S', -1],
                  [2, 'M', -1],
                  [2, 'M', 1],
                  [2, 'L', 1],
                  [2, 'L', 1],
                  [3, 'L', 1],
                  [3, 'M', 1],
                  [3, 'M', 1],
                  [3, 'L', 1],
                  [3, 'L', -1]])
    y = X[:, 2]
    X = X[:, :2]
    nb_classifier = NB_classifier()
    nb_classifier.fit(X, y)
    nb_classifier.predict([[2, 'S']])
