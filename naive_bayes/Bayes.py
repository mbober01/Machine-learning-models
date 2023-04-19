from utils import Utils, np, pd


def g_x(x, mean, std):
    result = 1 / (std * np.sqrt(2 * np.pi))
    result *= np.power(np.e, -(np.power(x - mean, 2)) / (2 * np.power(std, 2)))
    return result


class Bayes:
    def __init__(self):
        self.y = None
        self.X_train = None
        self.y_train = None
        self.cols = None
        self.mean_std = None

    def fit(self, train):
        self.y = train.columns[-1]
        self.X_train = train.drop(self.y, axis=1)
        self.y_train = train[self.y]
        self.cols = train[self.y].unique()

        self.mean_std = {
            col: [self.X_train[self.y_train == col].mean(),
                  self.X_train[self.y_train == col].std()]
            for col in self.cols
        }

    def predict(self, test):
        X_test = test.drop(self.y, axis=1)
        predictions = []
        for record in X_test.to_numpy():
            results = []
            for var in self.cols:
                result = 1
                for i, x in enumerate(record):
                    mean = self.mean_std[var][0][i]
                    std = self.mean_std[var][1][i]
                    result *= g_x(x, mean, std)
                results.append(result)
            predictions.append(self.cols[np.argmax(results)])
        return predictions

    def accuracy(self, test, pred):
        test = test[self.y]
        accuracy = 0
        for x, y in zip(test, pred):
            if x == y:
                accuracy += 1
        accuracy /= len(pred)
        accuracy *= 100
        return accuracy
