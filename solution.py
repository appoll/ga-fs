import numpy as np


class Solution:
    def __init__(self, features_count, features = None):
        """
        Wraps a binary vector of features' usage.
        Represents feature subset.
        :param features_count:
        """
        self.total_size = features_count
        # initialize one solution's binary vector as an array of zeros
        if features is None:
            self.features = np.zeros(self.total_size, dtype=bool)
        else:
            self.features = features

        # solution properties to be initialized with creation
        self.set_properties()

        # solution properties to be updated with evaluation
        self.fitness = -1
        self.duration = 0

    def random(self):
        # random binary numpy array
        self.features = np.random.randint(2, size=self.total_size, dtype=bool)
        # print 'Created solution of size %d' % len(self.features)
        # print 'Out of which true: %d' %sum(self.features)
        # print self.features
        self.set_properties()

        return self

    def to_string(self):
        feature_indices = np.nonzero(self.features)
        features_count = sum(self.features)
        ret = "Solution with %d features and fitness %f " % (features_count, self.fitness) + str(feature_indices)
        return ret

    def set_properties(self):
        """
        Sets the properties of a new solution, according to the feature subset characteristics
        """
        self.used_size = sum(self.features)
        indices = np.nonzero(self.features)
        self.used_features_indices = np.array_str(indices[0])
