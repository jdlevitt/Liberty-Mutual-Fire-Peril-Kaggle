######################################
#
# Kaggle Fire Insurance Competition
#
# Ridge model with top features
#
######################################

import numpy as np
import pandas as pd
import NW_gini_eval as gini


from sklearn import cross_validation
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn import svm
#from sklearn.ensemble import BaggingRegressor


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


#train = train.iloc[:,top_50]
#test = test.iloc[:,top_50]

importances = np.load("feature_importances1000.npy")
findices = np.argsort(importances)[::-1]
findices = np.array(findices).tolist()

#select cut-off for indices
feats = 40
findices = findices[0:feats]
tr = train.iloc[:,findices]
te = test.iloc[:,findices]



print "Data is preprocessed.  Starting algorithm..."

######################################
# Learning 
######################################

# setting up model

#clf = ensemble.RandomForestRegressor(n_estimators= 100, n_jobs = -1, verbose = 1)
#clf = BaggingRegressor(Ridge(), n_estimators= 20, max_samples = 0.5,  n_jobs = -1, verbose = 1)
#clf = BayesianRidge(alpha_1 = .1, alpha_2=.1)
clf = Ridge(alpha=1000)
	
	
# running algorithm on whole training set

clf.fit(tr, target.values)


#####################
# Printing eval of algorithm on training set:
#####################

print 'Now finding the evaluation...'
preds_train = clf.predict(tr)
eval = gini.normalized_weighted_gini(target, preds_train, variable.values)
print "The evaluation on the whole training set is:", eval

############################
# Output section
############################

# prediction

preds = clf.predict(te)
sample['target'] = preds

# print prediction to a file

sample.to_csv('submission_r40.csv', index = False)


print "Finished saving predictions to file."
