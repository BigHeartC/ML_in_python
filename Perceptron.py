import numpy as np


class perceptron(object):
    def __init__(self, w0=0, b0=0, eta=1, max_iter=100):
        self.w0 = w0
        self.b0 = b0
        self.eta = eta
        self.max_iter = max_iter

    def fit(self, X, y):
        # 将 b 拼接到 w 上
        self.w = np.append(np.ones(np.shape(X)[1]) * self.w0, self.b0)
        # 在 X 对应b的位置设为1
        X = np.column_stack((X, np.ones(np.shape(X)[0])))

        count, iter = 0, 0
        while True:
            iter += 1
            print('iter:', iter)
            for xi, yi in zip(X, y):
                if yi * (np.dot(self.w, xi)) <= 0:
                    self.w = np.add(self.w, np.dot(xi, self.eta * yi))
                    count = 0
                else:
                    count += 1

                if count >= len(X) or iter > self.max_iter:
                    return self.w

    def fit2(self, X, y):
        '''
        对偶形式
        :return: 
        '''
        X = np.array(X)
        y = np.array(y)
        alpha = np.zeros((len(X), 1))
        gram = np.matmul(X, X.transpose())
        count, iter = 0, 0
        print(gram)
        while True:
            iter += 1
            print('iter:', iter)
            for i in range(X.shape[0]):
                if y[i] * (sum(np.dot(alpha.transpose() * y, gram[i])) + self.b0) <= 0:
                    alpha[i] += self.eta
                    self.b0 += self.eta * y[i]
                    count = 0
                else:
                    count += 1

                if count >= len(X) or iter > self.max_iter:
                    self.w = np.append(np.dot(alpha.transpose() * y, X), self.b0)
                    return self.w


def predict(self, X):
    X = np.column_stack((X, np.ones(np.shape(X)[0])))
    return np.sign(np.dot(X, self.w))


if __name__ == '__main__':
    model = perceptron([0, 0], 0)
    print(model.fit2([[3, 3], [4, 3], [1, 1]], [1, 1, -1]))
    print('prediction:', model.predict([[-1, -1]]))
