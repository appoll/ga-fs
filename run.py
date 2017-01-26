import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from settings import DATASETS_PATH


class Run:
    def __init__(self, dataset=None):
        self.dataset = dataset

        # entire data in a pandas dataframe, including labels
        self.data = self.read_data()

    def read_data(self):
        data = pd.read_csv(DATASETS_PATH + self.dataset, delimiter=",", header=None)
        print("Reading %s dataset. Has shape:" % self.dataset)
        print data.shape
        return data

    def split(self, dataframe=None):
        """
        Splits data into training and testing data.
        :param dataframe: If not None, represents a per-column-slice of the data.
        :return:
        """
        if dataframe is None:
            data = self.data
        else:
            data = dataframe

        train_data, test_data = train_test_split(data, test_size=0.3, random_state=5)

        # labels from last column
        train_labels = train_data.iloc[:, -1]
        test_labels = test_data.iloc[:, -1]

        # drop last column
        train_data = train_data.iloc[:, :-1]
        test_data = test_data.iloc[:, :-1]

        return train_data, test_data, train_labels, test_labels

    def slice(self, selected):
        """
        Slices the data dataframe vertically according to the binary encoded features.
        Returns the train test split
        :param selected: Binary array of used features
        """
        # hack add one True to the selected features array so that the data_slice carries the label to the split method
        # append does not occur in place, new copy is created.
        sel = np.append(selected, [True])

        data_slice = self.data.loc[:, sel]
        data_slice = pd.DataFrame(data=data_slice)
        return self.split(dataframe=data_slice)

    def knn(self):
        train_data, test_data, train_labels, test_labels = self.split()

        knn_clf = KNeighborsClassifier(n_neighbors=2)
        knn_clf.fit(train_data, train_labels)

        knn_output = knn_clf.predict(test_data)
        score = accuracy_score(test_labels, knn_output)
        print '\nknn'
        print score
        return score

    def knn_sel(self, selected):
        """
        Perform knn on a slice of the data, corresponding to selected features.
        :param selected: Binary array encoding for selected features of the current run
        """
        train_data, test_data, train_labels, test_labels = self.slice(selected=selected)
        knn_clf = KNeighborsClassifier(n_neighbors=2)
        knn_clf.fit(train_data, train_labels)

        knn_output = knn_clf.predict(test_data)
        score = accuracy_score(test_labels, knn_output)
        # print '\nknn'
        # print score
        return score

    def log_reg(self):
        train_data, test_data, train_labels, test_labels = self.split()

        log_reg_clf = LogisticRegression()
        log_reg_clf.fit(train_data, train_labels)

        log_reg_output = log_reg_clf.predict(test_data)
        score = accuracy_score(test_labels, log_reg_output)
        print '\nlogistic regression'
        print score
        return score


if __name__ == '__main__':
    run = Run(dataset='ionosphere.data')
    run.knn()
    run.log_reg()
