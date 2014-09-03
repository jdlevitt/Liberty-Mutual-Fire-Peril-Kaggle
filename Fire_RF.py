######################################
#
# Kaggle Fire Insurance Competition
#
# Random Forest model with feature list
#
######################################

import numpy as np
import pandas as pd
import NW_gini_eval as gini

from sklearn import ensemble
from sklearn import cross_validation


########################
# Inputs
########################

print "Importing data sets..."
# input in batches for memory issue fix
train_batch = pd.read_csv('train_cat.csv', iterator = True, chunksize = 1000, engine = 'c')
train = pd.concat(list(train_batch), ignore_index = True)

test_batch = pd.read_csv('test_cat.csv', iterator = True, chunksize = 1000, engine = 'c')
test = pd.concat(list(test_batch), ignore_index = True)

sample = pd.read_csv('sampleSubmission.csv')

#useful columns which are NOT preprocessed
#target = pd.read_csv('train.csv', usecols=['target'])
#variable = pd.read_csv('train.csv', usecols=['var11'])
#target = target['target']
#variable = variable['var11']

#useful columns when data has not been normalized, etc.
target = train['target']
variable = train['var11']

#####
#For quick debugging
#####
#train = train.ix[0:10000,:]
#test = test.ix[0:10000,:]
#sample = sample.ix[0:10000,:]

#######################
# Preprocessing section
#######################
# (Currently this section has mostly been moved to a different file)

print "Data is read.  Starting preprocessing..."

# selecting our training data columns
train = train.ix[:,'var10':] # continuous data columns which we are currently using

test = test.ix[:,'var10':] # we have one less column in this data set


print "Data is preprocessed.  Starting algorithm..."

#######################
# Learning section
#######################

# setting up model

clf = ensemble.RandomForestRegressor(n_estimators= 1000, n_jobs = -1, verbose = 1)

# running algorithm on whole training set

clf.fit(train, target.values)

######################
# Feature Importance 
######################

importances = clf.feature_importances_

# create table of the most important
# code from scikit-learn.org example
indices = np.argsort(importances)[::-1]
print("Feature Ranking:")
for f in range(50):
	print("%d. feature %d (%f)" % (f+1, indices[f], importances[indices[f]]))
	
# save the feature importances to a numpy file for later reference

np.save("feature_importances.npy", importances)


#####################
# Printing eval of algorithm on training set:
#####################

print 'Now finding the evaluation...'
preds_train = clf.predict(train)
eval = gini.normalized_weighted_gini(target, preds_train, variable.values)
print "The evaluation on the whole training set is:", eval

############################
# Output section
############################

# prediction

preds = clf.predict(test)
sample['target'] = preds

# print prediction to a file

sample.to_csv('submission_rf.csv', index = False)


print "Finished saving predictions to file."
