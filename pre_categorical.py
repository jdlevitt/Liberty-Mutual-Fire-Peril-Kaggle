#######################################################
#
# Preprocessing for Kaggle Fire Insurance Challenge
#
# Fills NA, and min max scales continuous data columns
#
#######################################################


import numpy as np
import pandas as pd
from sklearn import preprocessing

print "Loading data..."
# input in batches for memory issue fix
train_batch = pd.read_csv('train.csv', iterator = True, chunksize = 1000, engine = 'c')
train = pd.concat(list(train_batch), ignore_index = True)
train = train.drop('dummy', 1) #get rid of non-predictive column

test_batch = pd.read_csv('test.csv', iterator = True, chunksize = 1000, engine = 'c')
test = pd.concat(list(test_batch), ignore_index = True)
test = test.drop('dummy', 1)

#####
#For quick debugging
#####
#train = train.iloc[0:2000,:]
#test = test.iloc[0:2000,:]

#####################################################


print "Data loaded, starting preprocessing..."

# preprocess

#fill with median from train and test averaged

train = train.fillna((train.median() + test.median())/2)
test = test.fillna((train.median() + test.median())/2)

# Now adding a save
train.to_csv('train_NA_filled.csv', index=None)
test.to_csv('test_NA_filled.csv', index=None)


print "NA values filled, dummifying categorical variables..."


# taking out columns 7 and 9 because of unknown labeling error
#train = train.drop(['var7', 'var9'],1)
#test = test.drop(['var7', 'var9'],1)

cat_cols = ['var1', 'var2', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8', 'var9']


for col in cat_cols:
	
	dummy_train = pd.get_dummies(train[col], prefix = col)
	dummy_train = dummy_train.ix[:,0:-1] #Need to eliminate the Z columns
	dummy_test = pd.get_dummies(test[col], prefix = col)
		
	#drop original column, put new columns at end of data frame. 
	train = train.drop(col, 1)
	test = test.drop(col, 1)
	train = train.join(dummy_train)
	test = test.join(dummy_test)
	
	print "%s is dummified." % col

###################################################

print "Data processed, saving to files..."

# write to a file

train.to_csv('train_cat.csv', index=None)
test.to_csv('test_cat.csv', index=None)