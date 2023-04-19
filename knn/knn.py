from utils import Utils, np, pd
from collections import Counter


class Knn:
    def __init__(self, train, test, labels, check, k=3, m=1):
        self.train = train
        self.test = test
        self.labels = labels
        self.check = check
        self.k = k
        self.m = m

    def train_model(self):
        types_list = []
        for record in self.test[self.labels].to_numpy():
            dist = []
            for train_record in self.train[self.labels].to_numpy():
                dist.append(Utils.distance(record, train_record, self.m))
            args = np.argsort(dist)
            variety = Counter(self.train.iloc[args[:self.k]]['variety']).most_common(1)[0][0]
            types_list.append(variety)
        return types_list

    def check_model(self, trained):
        percentage = 0
        for x, y in zip(trained, self.test[self.check]):
            if x == y:
                percentage += 1
        percentage /= len(trained)
        return percentage * 100