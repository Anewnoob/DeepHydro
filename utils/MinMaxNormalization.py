class MinMaxNormalization_01(object):
    '''MinMax Normalization --> [0, 1]
       x = (x - min) / (max - min).
    '''
    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = 1. * X * (self._max - self._min) + self._min
        return X