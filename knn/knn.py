from utils import Utils, np, pd,Counter
class Knn:
    def __init__(self, k=3, m=1):
        self.k = k
        self.m = m
        self.y = None
        self.X_train = None
        self.y_train = None
        self.cols = None
        self.train = None

    def fit(self, train,y=None):
        if y:
            self.y = y
        else:
            self.y = train.columns[-1]
        self.X_train = train.drop(self.y,axis=1)
        self.y_train = train[self.y]
        self.train = train

    def predict(self, test):
        X_test = test.drop(self.y, axis=1)
        predictions = []
        for record in X_test.to_numpy():
            dist = []
            for train_record in self.X_train.to_numpy():
                dist.append(Utils.distance(record,train_record,self.m))
            args = np.argsort(dist)
            variety = Counter(self.y_train.iloc[args[:self.k]]).most_common(1)[0][0]
            predictions.append(variety)
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
#%%
