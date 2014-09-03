Liberty-Mutual-Fire-Peril-Kaggle
================================

Code for my first Kaggle competition- 75th place out of >600 teams.

For my first Kaggle contest, I started simple by trying ridge regression on the basic numeric features and got pretty good results. To improve this, after encoding the categorical data, I ran a 1000 tree random forest regressor on most of the features using a 16x EC2 instance (took ~10 hours). This of course massively overfit the training set, but my goal was to rank the features, and then try some sort of backward feature selection.

I didn't have enough time (or money for the EC2 instances) to run more RFs using lots of trees and features, so I decided to run ridge regression on the top 40-70 features, and roughly tuned my alpha using a couple different K-fold cross-validations. This pointed to using about 60 features, but I felt I was still massively overfitting as I was getting a better public leaderboard score with 50 features. To make my model even more robust, I used 40 features for my final submission as time ran out.

One thing that I would have liked to have done is sub-sampled the non-positive training examples, as there were so few positives.

