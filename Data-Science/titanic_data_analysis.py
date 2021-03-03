#import relevant packages
import numpy as np
import pandas as pd
from matplotlib import pyplot

#import dataset
TitanicDS_train = pd.read_csv("~/Desktop/Involvement/CodeHub/datasets/titanic/train.csv")
TitanicDS_train.fillna("NA")
TitanicDS_test = pd.read_csv("~/Desktop/Involvement/CodeHub/datasets/titanic/test.csv")

#EXPLORATORY ANALYSIS

#Feature definitions:

#1. EXPLANATORY:
#PassengerID - identifier
#Pclass - ticket class (key: 1, 2, or 3) (int)
#Name - full name (String)
#Sex - sex of passenger (String, male or female)
#Age - passenger age in years (int)
#SibSp - # siblings / spouses aboard (int)
#parch - # parents / children aboard (int)
#ticket - ticket number (some inconsistency in format)
#fare - passenger fare (float)
#cabin - cabin number (some inconsistency in format)
#embarked - Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

#RESPONSE:
#survived - 0: died, 1: survived


#features of interest:
#"pclass", "sex", "Age", "sibsp", "parch", "fare", "embarked"
#quantitative: age, sibsp, parch, fare
#qualitative: pclass, sex, embarked


#1a. QUANTITATIVE ANALYSIS

#1a1) sample statistics
interest_features_quant = ["Age", "SibSp", "Parch", "Fare"]

print("Sample Statistics:")
def printstats (dataset):
    for i in range(0,4):
        print(interest_features_quant[i], ":")
        print("  min: ", min(dataset[interest_features_quant[i]]))
        print("  max: ", max(dataset[interest_features_quant[i]]))
        print("  mean: ", np.nanmean(dataset[interest_features_quant[i]])) #calculate mean ignoring missing values
        print("  std dev: ", np.nanstd(dataset[interest_features_quant[i]])) #calculate sd ignoring missing values
        print()
#end of printstats method

printstats(TitanicDS_train)


#1a2) sample statistics, survived vs died

#filter the data
TitanicDS_train_survived = TitanicDS_train[TitanicDS_train["Survived"] == 1]
TitanicDS_train_died = TitanicDS_train[TitanicDS_train["Survived"] == 0]

#use same method as above to show sample statistics
print("Sample Statistics - Survived:")
printstats(TitanicDS_train_survived)
print("Sample Statistics - Died:")
printstats(TitanicDS_train_died)


#1a3) plotting variable distributions
#died vs survived
def histograms (survived, died):
    for i in range(0,4):
        pyplot.hist(survived[interest_features_quant[i]], alpha=0.5, label='survived')
        pyplot.hist(died[interest_features_quant[i]], alpha=0.5, label='died')
        pyplot.title("{} Distribution".format(interest_features_quant[i]))
        pyplot.legend(loc='upper right')
        pyplot.show()

histograms(TitanicDS_train_survived, TitanicDS_train_died)


#1b. CATEGORICAL ANALYSIS

interest_features_qual = ["Pclass", "Sex", "Embarked"]

#barplots: died vs survived

#Pclass
#filter data
TitanicDS_Pc1 = TitanicDS_train[TitanicDS_train["Pclass"] == 1]
TitanicDS_Pc2 = TitanicDS_train[TitanicDS_train["Pclass"] == 2]
TitanicDS_Pc3 = TitanicDS_train[TitanicDS_train["Pclass"] == 3]

pyplot.bar(x = [0, 1], height = np.bincount(TitanicDS_Pc1["Survived"]))
pyplot.title("Survival for Pclass = 1")
pyplot.show()

pyplot.bar(x = [0, 1], height = np.bincount(TitanicDS_Pc2["Survived"]))
pyplot.title("Survival for Pclass = 2")
pyplot.show()

pyplot.bar(x = [0, 1], height = np.bincount(TitanicDS_Pc3["Survived"]))
pyplot.title("Survival for Pclass = 3")
pyplot.show()

#clealy significant difference bwtn class and survival rates :: important for 
#variable selection; likely a significant predictor

#Sex
TitanicDS_F = TitanicDS_train[TitanicDS_train["Sex"] == "female"]
TitanicDS_M = TitanicDS_train[TitanicDS_train["Sex"] == "male"]

pyplot.bar(x = [0, 1], height = np.bincount(TitanicDS_F["Survived"]))
pyplot.title("Survival for Sex = Female")
pyplot.show()

pyplot.bar(x = [0, 1], height = np.bincount(TitanicDS_M["Survived"]))
pyplot.title("Survival for Sex = Male")
pyplot.show()

#VERY SIGNIFICANT!!!

#Embarked
TitanicDS_S = TitanicDS_train[TitanicDS_train["Embarked"] == "S"]
TitanicDS_C = TitanicDS_train[TitanicDS_train["Embarked"] == "C"]
TitanicDS_Q = TitanicDS_train[TitanicDS_train["Embarked"] == "Q"]

pyplot.bar(x = [0, 1], height = np.bincount(TitanicDS_S["Survived"]))
pyplot.title("Survival for Embarked = S")
pyplot.show()

pyplot.bar(x = [0, 1], height = np.bincount(TitanicDS_C["Survived"]))
pyplot.title("Survival for Embarked = C")
pyplot.show()

pyplot.bar(x = [0, 1], height = np.bincount(TitanicDS_Q["Survived"]))
pyplot.title("Survival for Embarked = Q")
pyplot.show()

#appears to be a significant difference (C, more people survived)
#is embarked related multicolinear with PClass?


#2. FEATURE SELECTION
#correlation matrix
#only applicable for quantitative data
print("correlation matrix:")
print(TitanicDS_train.corr())
print(TitanicDS_train.corr()["Survived"])
#highest correlations (quanitative data only): 1) Pclass (actually categorical with int), 
#2) Fare (related to Pclass), 3) Parch (surprising), 4) age

#significant categorical data: 
#definitely sex and PClass, potentially Embarked

#NOTE: can also observe relationship between quantitative variables by plotting w scatterplots


#3. MODEL DEVELOPMENT
#possible models:
    #unsupervised: determines patterns from unknown repsonse labels
        #ex. clustering
    #supervised: determines patterns and knows response label 
        #almost everything else ex. regression (Simple, multiple, logistic, etc.), 
        #classifiers (kNN, Naive Bayes), decision trees,  
    #eventually: neural networks, deep learning
#goal: best possible at making accurate predictions for TESTING dataset

#NOTE: in the test dataset provided online, the response is not included (this is 
#probably since the dataset is intended for a competition); instead, split the 
#current training dataset randomly into training and testing subsets to evaluate
#the effectiveness of various models

#!!!!!!!!!!!!!!!!!!!!!!!!! split into testing and training here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


#3.1 - kNN (k-nearest-neighbors model)
#uses labels of nearby points (the number of nearby points = k) to predict the 
#label at a given point

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

Features = ['Age', 'Sex', 'Pclass']
TitanicDS_train_X = TitanicDS_train[Features].fillna(-1)
TitanicDS_train_Y = TitanicDS_train["Survived"].fillna(-1) #must provide value for na for functionality of models
#model has trouble interpreting string values for sex :: must encode with numbers (ex 0 and 1)
TitanicDS_train_X = pd.get_dummies(TitanicDS_train_X, columns=['Sex'])
TitanicDS_train_X = TitanicDS_train_X.drop(labels = "Sex_male", axis = 1)

#model with k=3 neighbors
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(TitanicDS_train_X, TitanicDS_train_Y)
survival_pred_Y = classifier.predict(TitanicDS_train_X)
#comparing observed and predicted
print("accuracy:", metrics.accuracy_score(TitanicDS_train_Y, survival_pred_Y))
print(classification_report(TitanicDS_train_Y, survival_pred_Y))
