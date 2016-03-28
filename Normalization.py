
import numpy as np
from sklearn import cross_validation
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
     
from sklearn.datasets import load_svmlight_file
X , y = load_svmlight_file("D:\Machine Learning\Assignment 3\scop_motif.data")
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=4)

#normalization
norm_data=preprocessing.normalize(X, norm='l2')
X = norm_data

#print cross_validation.cross_val_score(classifier, X, y, cv=5, scoring='accuracy')
#print np.mean(cross_validation.cross_val_score(classifier, X, y, cv=5, scoring='accuracy'))

classifier = svm.SVC(kernel='rbf').fit(norm_data, y)
#print classifier.score(X_test, y_test)

#print cross_validation.cross_val_score(classifier, X, y, cv=5, scoring='roc_auc')
#m = cross_validation.cross_val_score(classifier, X, y, cv=5, scoring='roc_auc')

#OR
#cv = cross_validation.StratifiedKFold(y, 5)
#print cross_validation.cross_val_score(classifier, X, y, cv=cv, scoring='roc_auc')

"""
data=np.genfromtxt("D:\Machine Learning\Assignment 3\scop_motif.data")
X=data[:,1:]
y=data[:,0]
"""

from sklearn.grid_search import GridSearchCV
#Cs = np.logspace(-2, 3, 6)
#classifier = GridSearchCV(estimator=svm.LinearSVC(), param_grid=dict(C=Cs) )
#classifier.fit(X, y)

"""
print classifier.best_score_
print classifier.best_estimator_

"""
print cross_validation.cross_val_score(classifier, X, y, cv=5)

param_grid_gaussian = [
#  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['poly'], 'degree':[1,2,3,4,5]}]
  {'C': [1, 10, 100, 1000], 'gamma': [0.01, 0.1, 1, 10, 100], 'kernel': ['rbf']}]
 
classifier = GridSearchCV(estimator=svm.SVC(), param_grid=param_grid_gaussian, cv=5, scoring='roc_auc')
classifier.fit(X, y)
print "grid scores"
print classifier.grid_scores_
print "cross validation"
print cross_validation.cross_val_score(classifier, X, y, cv=5, scoring='roc_auc')
#print np.mean(cross_validation.cross_val_score(classifier, X, y, cv=5, scoring='roc_auc'))
print "classifier.best_params_"
print classifier.best_params_
