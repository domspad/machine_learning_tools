#!/usr/bin/python
#title			:one_model_feat.py
#description	:given a set of features and one model, fits model on all combinations of feats and outputs AMS scores and which feats used in log file
#author			:domspad
#date			:20140903
#version		:0.1
#usage			:python one_model_feat.py
#notes			:
#python_version	:2.6.6
#==============================================================================

import pandas as pd 
import numpy as np 
from math import sqrt, log
import higgs_tools.calc_ams as ams
import operator #to help sort dictionaries
import random #in case classifier cant take all the data
import itertools #to iterate over combos of features

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import ParameterGrid
# from sklearn.metrics import precision_score, accuracy_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
# from sklearn import grid_search
# from sklearn.cross_validation import cross_val_score


def label_01(label) :
    if label == 's' :
        return 1
    if label == 'b' :
        return 0
    else :
        assert 0

def label_sb(label) :
    if label == 1 :
        return 's'
    if label == 0 :
        return 'b'
    else :
        assert 0

"""
SETTINGS
"""

SUBCOLL = [0,1,2,13,11,10,6,7,5,19] #given by indices in range [0,29]
SIZES = [5,6,7,8,9,10] #size of feature groupings to try (NOTE '1' will try ALL feats indvid)
MODEL = 'randfor'
PARAMS = {
		'n_estimators':			40,
		#criterion:				'gini', 
		# max_depth:				None, 	
		# min_samples_split:		2, 
		'min_samples_leaf':		100, 
		# 'max_features':			[None,'auto',0.5],
		# bootstrap:				True, 
		# oob_score:				False, 
		# n_jobs:					1, 
		# random_state:			None, 
		'verbose':				1, 
		# min_density:			None, 
		# compute_importances:	None
		}

NAS = 'imputer_mean'  # drop, impute_mean, impute_median, fill_0,...
SCALE = 'standard'  #standard...
FOLDS = 3
VERBOSE = True
MILDVERBOSE = True #lets you know what test your on out of what

SMALLER_DATA = (True, 150000) #if classifier can handle more than 100,000 samples

"""
DATA
"""

DATAFILE = '/Users/dominicspadacene/Desktop/higgs_challenge/data/original_data/training.csv'
datadf = pd.read_csv(DATAFILE,na_values=-999.0,index_col='EventId')
datadf['Label'] = datadf.Label.apply(label_01)

if SMALLER_DATA[0] :
	if (VERBOSE): print 'random cut to data bc too big for classifier'
	rows = random.sample(datadf.index, SMALLER_DATA[1])
	datadf = datadf.ix[rows]

FEATS = datadf.columns.tolist()[:-2]
LABEL = datadf.columns.tolist()[-1]

if 'impute' in NAS :
	if (VERBOSE): print NAS
	kind = NAS[NAS.rindex('_')+1:]
	X = datadf[FEATS].values
	y = datadf[LABEL].values
	imputer = Imputer(strategy=kind,axis=1)
	X = imputer.fit_transform(X,y)
elif NAS == 'drop' : 
	if (VERBOSE): print 'dropping Nans'
	datadf = datadf.dropna()
	X = datadf[FEATS].values
	y = datadf[LABEL].values
elif NAS == 'fill_0' :
	if (VERBOSE): print 'filling with 0\'s'
	datadf = datadf.fillna(0)
	X = datadf[FEATS].values
	y = datadf[LABEL].values
elif SCALE == 'standard' : #nothing...
	if (VERBOSE): print 'dropping Nans, because standardize cant handle NaN'
	datadf = datadf.dropna()
	X = datadf[FEATS].values
	y = datadf[LABEL].values
else :
    if (VERBOSE): print 'nothing doing'
    X = datadf[FEATS].values
    y = datadf[LABEL].values

if SCALE == 'standard' :
	if (VERBOSE): print 'standardizing data'
	scaler = StandardScaler()
	X = scaler.fit_transform(X,y)

if (VERBOSE): print 'generating the {} folds'.format(FOLDS)
foldindexes = StratifiedKFold(y, n_folds=FOLDS)

"""
MODEL
"""
if VERBOSE : 'generating model {mod}'.format(mod=MODEL)
if MODEL == 'sgd' :
	from sklearn.linear_model import SGDClassifier
	model = SGDClassifier(**PARAMS)
elif MODEL == 'randfor' :
	from sklearn.ensemble import RandomForestClassifier
	model = RandomForestClassifier(**PARAMS)

"""
LOOP
"""
results = {}
header_str = '{ams},{model},{feats}'.format(ams='AMS',
											model='model_type',
											feats=','.join(FEATS))
results[header_str] = 10

SUBSETS = [] #set of indexsets, each indexset being a set of feats to try together
for size in SIZES :
	if size == 1 :
		SUBSETS += list(itertools.combinations(range(len(FEATS)),1))
	elif size in range(2,len(SUBCOLL)+1):
		SUBSETS += list(itertools.combinations(SUBCOLL, size))
	else :
		assert 0
NUMBER_TESTS = len(SUBSETS)

if MILDVERBOSE : print 'starting the {} tests'.format(NUMBER_TESTS)
for setnum, featset in enumerate(SUBSETS) :
	if MILDVERBOSE : print 'TEST {} of {}'.format(setnum, NUMBER_TESTS)
	featset = list(featset)
	allams = []
	for num, (train, test) in enumerate(foldindexes) :

		model.fit(X[train][:,featset], y[train])
		preds = model.predict(X[test][:,featset])

		#calc AMS
		amsdf = datadf.iloc[test,][['Weight','Label']]
		amsdf['Prediction'] = preds
		amsdf['Label'] = amsdf.Label.apply(label_sb)
		amsdf['Prediction'] = amsdf.Prediction.apply(label_sb)
		sb = ams.calc_sb(amsdf)
		AMS = ams.AMS(sb[0],sb[1])
		if (VERBOSE): print 'AMS = {:<20}'.format(AMS)
		allams.append(AMS)

	avgAMS = np.array(allams).mean()
	feats = [str(int(i in featset)) for i in range(30)]
	outputline = '{ams},{model},{feats}'.format(ams=avgAMS,
												model=MODEL,
												feats=','.join(feats))
	results[outputline] = avgAMS

"""
print
"""
newout = ''
sorted_results = sorted(results.iteritems(),
						key=operator.itemgetter(1),
						reverse=True)
for string, ams in sorted_results :
	newout += '{out}\n'.format(out=string)

with open('models.log','r+') as log :
	content = log.read()
	log.seek(0,0)
	log.write(newout.rstrip('\r\n')+'\n'+content)





