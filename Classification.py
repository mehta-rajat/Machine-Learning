# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 02:05:33 2021

@author: rajat
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OrdinalEncoder

class Classification:
    x_train = pd.DataFrame()
    y_train = pd.DataFrame()
    x_test = pd.DataFrame()
    y_test = pd.DataFrame()
    
    def __init__(self, x , y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
    
    def data_cleaning(self):
        pass
    
    def logistic_regression(self):
        clf = LogisticRegression(random_state=0).fit(self.x_train, self.y_train)
        return clf
    
    def naive_bayes(self):
        clf = MultinomialNB()
        clf.fit(self.x_train, self.y_train)
        return clf
    
    def cart_decision_tree(self):
        clf = DecisionTreeClassifier(min_samples_leaf = 4,random_state=0)

        clf = clf.fit(self.x_train,self.y_train)
        return clf
    
    def gradient_boosting(self):
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
        clf = clf.fit(self.x_train, self.y_train)
        return clf
    
    def random_forest(self):
        clf = RandomForestClassifier(max_depth= None, random_state=0)

        clf = clf.fit(self.x_train,self.y_train)
        return clf
    
    def evaluate_test(x_test,y_test,clf_model):
        y_pred = clf_model.predict(x_test)
        confusion_matrix_var = confusion_matrix(y_test, y_pred) 
        return ((confusion_matrix_var[0][1] + confusion_matrix_var[1][0]) * 100)/ (confusion_matrix_var[0][1] + confusion_matrix_var[1][0] + confusion_matrix_var[0][0] + confusion_matrix_var[1][1])
        
    
    def evaluate(self,clf_model):
        y_pred = clf_model.predict(self.x_test)
        confusion_matrix_var = confusion_matrix(self.y_test, y_pred) 
        return ((confusion_matrix_var[0][1] + confusion_matrix_var[1][0]) * 100)/ (confusion_matrix_var[0][1] + confusion_matrix_var[1][0] + confusion_matrix_var[0][0] + confusion_matrix_var[1][1])
        
    
    def best_results_test(self):
        clf1 = self.logistic_regression()
        clf2 = self.naive_bayes()
        clf3 = self.cart_decision_tree()
        clf4 = self.gradient_boosting()
        clf5 = self.random_forest()
        clf = [clf1,clf2,clf3,clf4,clf5]
        clf_name = ["LogisticRegression","NaiveBayes", "CART", "GBM", "RandomForest"]
        results = []
        for clf_m in clf:
            results.append(self.evaluate(clf_m))
        
        print("The best model is :", clf_name[results.argmin()], "with error rate")
        
    def main():
        data = pd.read_csv("customer_personality_dataset.csv")
        #print(data.columns)
        X = data[["Education", "Marital_Status", "Income", "Kidhome", "NumDealsPurchases",	"NumWebPurchases",	"NumCatalogPurchases",	"NumStorePurchases",	"NumWebVisitsMonth"]]
        Y = data["Response"]
        ordinalencoder = OrdinalEncoder()

        X1 = pd.DataFrame()
        X1 = ordinalencoder.fit_transform(X.astype(str))
        X1 = pd.DataFrame(data = X1,columns = X.columns)
        #print(X1)
        
        classification = Classification(X1,Y)
        clf = classification.random_forest()
        result = classification.evaluate_test(X1, Y, clf)
        print(result)
        #print(classification.x_test)
    
    if __name__ == "__main__":
        main()        
        
    
    