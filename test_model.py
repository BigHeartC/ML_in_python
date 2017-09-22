from sklearn import datasets
from Perceptron import perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def test_perceptron():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X = X[y < 2]
    y = y[y < 2]
    y[y == 0] = -1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = perceptron()
    print(model.fit(X_train, y_train))
    y_pred = model.predict(X_test)
    print('准确率: {}%'.format(accuracy_score(y_test, y_pred) * 100))


if __name__ == '__main__':
    test_perceptron()

