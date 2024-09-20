import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
#import matplotlib.pyplot as plt
import argparse

class MyLogisticRegression:
    def __init__(self, dataset_num, perform_test):
        self.training_set = None
        self.test_set = None
        self.model_logistic = None
        self.model_linear = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.perform_test = perform_test
        self.dataset_num = dataset_num
        self.read_csv(self.dataset_num)

    def read_csv(self, dataset_num):
        if dataset_num == '1':
            train_dataset_file = 'train_q1_1.csv'
            test_dataset_file = 'test_q1_1.csv'
        elif dataset_num == '2':
            train_dataset_file = 'train_q1_2.csv'
            test_dataset_file = 'test_q1_2.csv'
        else:
            print("Unsupported dataset number")

        self.training_set = pd.read_csv(train_dataset_file)
        self.X_train = self.training_set[['exam_score_1', 'exam_score_2']].values
        self.y_train = self.training_set['label'].values

        if self.perform_test:
            self.test_set = pd.read_csv(test_dataset_file)
            self.X_test = self.test_set[['exam_score_1', 'exam_score_2']].values
            self.y_test = self.test_set['label'].values

    def model_fit_linear(self):
        '''Initialize the linear regression model and fit it to the training data.'''
        self.model_linear = LinearRegression()
        self.model_linear.fit(self.X_train, self.y_train)

    def model_fit_logistic(self):
        '''Initialize the logistic regression model and fit it to the training data.'''
        self.model_logistic = LogisticRegression()
        self.model_logistic.fit(self.X_train, self.y_train)

    def model_predict_linear(self):
        '''Calculate and return the accuracy, precision, recall, f1, support of the linear model.'''
        self.model_fit_linear()

        y_pred_train = self.model_linear.predict(self.X_train)
        y_pred_train_binary = (y_pred_train >= 0.5).astype(int)

        if self.X_test is not None:
            y_pred_test = self.model_linear.predict(self.X_test)
            y_pred_test_binary = (y_pred_test >= 0.5).astype(int)
            accuracy = accuracy_score(self.y_test, y_pred_test_binary)
            precision, recall, f1, support = precision_recall_fscore_support(self.y_test, y_pred_test_binary)
        else:
            accuracy = accuracy_score(self.y_train, y_pred_train_binary)
            precision, recall, f1, support = precision_recall_fscore_support(self.y_train, y_pred_train_binary)

        return [accuracy, precision, recall, f1, support]

    def model_predict_logistic(self):
        '''Calculate and return the accuracy, precision, recall, f1, support of the logistic model.'''
        self.model_fit_logistic()

        if self.X_test is not None:
            y_pred_test = self.model_logistic.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred_test)
            precision, recall, f1, support = precision_recall_fscore_support(self.y_test, y_pred_test)
        else:
            y_pred_train = self.model_logistic.predict(self.X_train)
            accuracy = accuracy_score(self.y_train, y_pred_train)
            precision, recall, f1, support = precision_recall_fscore_support(self.y_train, y_pred_train)

        return [accuracy, precision, recall, f1, support]
    '''
    def plot_decision_boundary(self, model, title, filename):
        #Plots decision boundaries of linear or logistic regression models.
        # mesh grid
        x_min, x_max = self.X_train[:, 0].min() - 1, self.X_train[:, 0].max() + 1
        y_min, y_max = self.X_train[:, 1].min() - 1, self.X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        Z = model.predict(grid_points)
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, edgecolors='k', marker='o', s=30, label='Train')
        if self.X_test is not None:
            plt.scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.y_test, edgecolors='k', marker='x', s=50, label='Test')
        plt.title(title)
        plt.xlabel('Exam Score 1')
        plt.ylabel('Exam Score 2')
        plt.legend()
        plt.savefig(filename)
        plt.clf()
    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Regression')
    parser.add_argument('-d','--dataset_num', type=str, default = "1", choices=["1","2"], help='string indicating datset number. For example, 1 or 2')
    parser.add_argument('-t','--perform_test', action='store_true', help='boolean to indicate inference')
    args = parser.parse_args()

    classifier = MyLogisticRegression(args.dataset_num, args.perform_test)
    acc = classifier.model_predict_linear()
    acc = classifier.model_predict_logistic()

    # Linear regression
    #classifier.model_fit_linear()
    #linear_filename = f'linear_regression_{args.dataset_num}.png'
    #classifier.plot_decision_boundary(classifier.model_linear, "Linear Regression Decision Boundary", linear_filename)

    # Logistic regression
    #classifier.model_fit_logistic()
    #logistic_filename = f'logistic_regression_{args.dataset_num}.png'
    #classifier.plot_decision_boundary(classifier.model_logistic, "Logistic Regression Decision Boundary", logistic_filename)

    