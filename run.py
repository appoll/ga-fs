from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from settings import DATASETS_PATH
from utils import dtime_to_seconds


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

    def split(self, seed, dataframe=None):
        """
        Splits data into training and testing data.
        :param dataframe: If not None, represents a per-column-slice of the data.
        :return:
        """
        if dataframe is None:
            data = self.data
        else:
            data = dataframe

        train_data, test_data = train_test_split(data, test_size=0.3, random_state=seed)

        # labels from last column
        train_labels = train_data.iloc[:, -1]
        test_labels = test_data.iloc[:, -1]

        # drop last column
        train_data = train_data.iloc[:, :-1]
        test_data = test_data.iloc[:, :-1]

        return train_data, test_data, train_labels, test_labels

    def slice(self, selected, seed):
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
        return self.split(dataframe=data_slice, seed = seed)

    def knn_cv(self, splits, seed):
        # labels from last column
        train_labels = self.data.iloc[:, -1]
        # drop last column
        train_data = self.data.iloc[:, :-1]

        start = datetime.now()
        knn_clf = KNeighborsClassifier(n_neighbors=2)
        cv_generator = ShuffleSplit(n_splits=splits, test_size=0.3, random_state=seed)
        scores = cross_val_score(knn_clf, train_data, train_labels, cv=cv_generator)
        end = datetime.now()

        print 'KNN %d fold mean accuracy: ' % splits
        print scores
        print np.mean(scores)
        # returning the average of cross validation runs will result in the threshold being below the randomly generated solutions in the first generation already
        return scores
        # return np.max(scores), dtime_to_seconds(end - start) / splits
        # return np.mean(scores), dtime_to_seconds(end - start) / splits


    def knn_sel(self, selected, seed):
        """
        Perform knn on a slice of the data, corresponding to selected features.
        :param selected: Binary array encoding for selected features of the current run
        """
        train_data, test_data, train_labels, test_labels = self.slice(selected=selected, seed=seed)

        start = datetime.now()
        knn_clf = KNeighborsClassifier(n_neighbors=2)
        knn_clf.fit(train_data, train_labels)

        knn_output = knn_clf.predict(test_data)
        score = accuracy_score(test_labels, knn_output)
        end = datetime.now()

        # print '\nknn'
        # print score
        return score, dtime_to_seconds(end - start)


    def knn_sel_cv(self, selected, seed):
        """
        Perform knn on a slice of the data, corresponding to selected features.
        :param selected: Binary array encoding for selected features of the current run
        """
        sel = np.append(selected, [True])

        data_slice = self.data.loc[:, sel]
        data_slice = pd.DataFrame(data=data_slice)

        # labels from last column
        train_labels = data_slice.iloc[:, -1]
        # drop last column
        train_data = data_slice.iloc[:, :-1]

        start = datetime.now()
        knn_clf = KNeighborsClassifier(n_neighbors=2)
        cv_generator = ShuffleSplit(n_splits=10, test_size=0.3, random_state=seed)
        scores = cross_val_score(knn_clf, train_data, train_labels, cv=cv_generator)
        end = datetime.now()

        # print '\nknn'
        # print scores
        # return scores
        # return scores
        return np.mean(scores), dtime_to_seconds(end - start)


    def knn_validate_sub(self, selected, seed):
        """
        Used to validate a subset of features
        :param selected:
        :param seed:
        :return:
        """
        train_data, test_data, train_labels, test_labels = self.slice(selected=selected, seed=seed)

        knn_clf = KNeighborsClassifier(n_neighbors=2)
        knn_clf.fit(train_data, train_labels)

        knn_output = knn_clf.predict(test_data)
        clf_report = classification_report(test_labels, knn_output)
        cnf_m = confusion_matrix(test_labels, knn_output)
        return clf_report, cnf_m

    def knn_validate_full(self, seed):
        """
        Used to validate the entire feature set
        :param seed:
        :return:
        """
        train_data, test_data, train_labels, test_labels = self.split(seed=seed)

        start = datetime.now()
        knn_clf = KNeighborsClassifier(n_neighbors=2)
        knn_clf.fit(train_data, train_labels)

        knn_output = knn_clf.predict(test_data)
        clf_report = classification_report(test_labels, knn_output)
        cnf_m = confusion_matrix(test_labels, knn_output)
        return clf_report, cnf_m


    def svm_sel(self, selected):
        """
        Perform knn on a slice of the data, corresponding to selected features.
        :param selected: Binary array encoding for selected features of the current run
        """
        train_data, test_data, train_labels, test_labels = self.slice(selected=selected)
        svm_clf = SVC(kernel='linear')
        svm_clf.fit(train_data, train_labels)

        svm_output = svm_clf.predict(test_data)
        score = accuracy_score(test_labels, svm_output)
        # print '\n svm'
        # print score
        return score

    def log_reg(self):
        train_data, test_data, train_labels, test_labels = self.split(5)

        log_reg_clf = LogisticRegression()
        log_reg_clf.fit(train_data, train_labels)

        log_reg_output = log_reg_clf.predict(test_data)
        score = accuracy_score(test_labels, log_reg_output)
        print '\nlogistic regression'
        print score
        return score

    # def univariate_fs(self):
    #     # labels from last column
    #     train_labels = self.data.iloc[:, -1]
    #     # drop last column
    #     train_data = self.data.iloc[:, :-1]
    #
    #     best_features = SelectKBest(f_classif, k=13).fit_transform(X=train_data, y=train_labels)
    #     return best_features[0]

if __name__ == '__main__':
    run = Run(dataset='ionosphere.data')
    score, duration = run.knn_validate_full()
    cv_scores, cv_duration = run.knn_cv(10)
    # run.log_reg()
    print score
    print duration

    print cv_scores
    print np.mean(cv_scores)
    print cv_duration

    print run.univariate_fs()