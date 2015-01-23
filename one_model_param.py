#!/usr/bin/python
#title			:one_model_param.py
#description	:my own gridsearch + kfold cv implementation of optimizing the params of one model (feature selection fixed)
#author			:domspad
#date			:20140828
#version		:0.1
#usage			:python one_model_param.py
#notes			:
#python_version	:2.6.6
#==============================================================================

"""

IN:			sklearn model
			parameter grid for model
			features to use (FIXED)
			flags on data cleaning (nas, and standardize)

OUT: 		print out of all model results (AMS and rough AMS, averaged over K cross_validated cuts of data)
				for each parameter setting which is appended to the models.log file
				write out the feature importances in a separate file: models_feat.log file 


CHECK: 		1) different results from sklearn's gridcvsearch 
			2) different ams rankings from rough_ams 
			3) Obv: what paramters for a given model are best (and how much variance there is in ams results)
					models: 		SGD classifier 		clf.coef_ gives array of coefs)
									Linear svc 			clf.coef_ "")
									RBF Svm?			clf.coef_
									Naive Bayes			   ?
									LDA? QDA? Disc Analy. ?
									NNeighbors			NONE)
									DecisionTreeClassi	clf.feature_importances_ gives Gini importance of features)
									Gradient Tree Boost clf.feature_importances_
									RandomForest		clf.feature_importances_
									AdaBoost			clf.feature_importances_
"""
import pandas as pd 
import numpy as np 
from math import sqrt, log
import higgs_tools.calc_ams as ams
import operator #to help sort dictionaries
import random #in case classifier can take all the data

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import ParameterGrid
# from sklearn.metrics import precision_score, accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
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
SET params 

ex:  for sgd_classifier

{'alpha': 0.0001,
 'class_weight': None,
 'epsilon': 0.1,
 'eta0': 0.0,
 'fit_intercept': True,
 'l1_ratio': 0.15,
 'learning_rate': 'optimal',
 'loss': 'hinge',
 'n_iter': 5,
 'n_jobs': 1,
 'penalty': 'l2',
 'power_t': 0.5,
 'random_state': None,
 'rho': None,
 'shuffle': False,
 'verbose': 0,
 'warm_start': False}

"""
MODEL = 'randfor'
NAS = 'fill_0'  # drop, impute_mean, impute_median, fill_0,...
SCALE = 'standard'  #standard...
FOLDS = 3
VERBOSE = True
MILDVERBOSE = True #lets you know what test your on out of what

SMALLER_DATA = (False, 30000) #if classifier cant handle more than 100,000 samples

param_ranges =  {
				'n_estimators':			[20,40,80],
				#criterion:				'gini', 
				# max_depth:				None, 	
				'min_samples_split':		[1,10], 
				'min_samples_leaf':		[1,10,100], 
				'max_features':			[None,'sqrt'],
				# bootstrap:				True, 
				# oob_score:				False, 
				# n_jobs:					1, 
				# random_state:			None, 
				'verbose':				[1], 
				# min_density:			None, 
				# compute_importances:	None
				}
"""
LOAD Data

	Nas
	Stand+Norm

"""
DATAFILE = '/Users/dominicspadacene/Desktop/higgs_challenge/data/data_slices/train__remainhigg.csv'
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
"""
Model loop

	Data kFold loop on folds

		fit

		predict

		calc_AMS
		calc rough_ams

	report_model
"""
results = {}
header_str = '{ams:<15} {rams:<15} {model:<15} {datasize:<10} {nas:<15} {stand:<15} {folds:<10} {numf:<10} {params:<20}'.format(
				ams='AMS',
				rams='roughAMS',
				model='Model',
				datasize='data_size',
				nas='NAs',
				stand='Stand',
				folds='Kfold',
				numf='NumFeat',
				params='Params')
results[header_str] = 10 #Just so that its always first printed...
if (VERBOSE): print 'generating the {} folds'.format(FOLDS)
foldindexes = StratifiedKFold(y, n_folds=FOLDS)

modeltests = ParameterGrid(param_ranges)
for modelnum, model_params in enumerate(modeltests) :
	if MILDVERBOSE : print 'TEST {modelnum} out of {total}\n'.format(modelnum=modelnum, total=len(modeltests)-1)
	if (VERBOSE): print 'preparing the {mod} model with {pms} params'.format(mod=MODEL,
																			 pms=model_params)
	if MODEL == 'sgd' :
		model = SGDClassifier(**model_params)
	elif MODEL == 'svc' :
		model = SVC(**model_params)
	elif MODEL == 'randfor' :
		model = RandomForestClassifier(**model_params)
	elif MODEL == 'adaboo' :
		from sklearn.ensemble import AdaBoostClassifier
		model = AdaBoostClassifier(**model_params)

	allams = []
	allrough = []
	for num, (train, test) in enumerate(foldindexes) :
		if (VERBOSE): print 'test {num} of {foldnum}'.format(num=num, foldnum=FOLDS)
		model.fit(X[train], y[train])
		preds = model.predict(X[test])

		roughAMS = ams.my_rough_AMS(y[test], preds)
		allrough.append(roughAMS)
		# calc AMS
		amsdf = datadf.iloc[test,][['Weight','Label']]
		amsdf['Prediction'] = preds
		amsdf['Label'] = amsdf.Label.apply(label_sb)
		amsdf['Prediction'] = amsdf.Prediction.apply(label_sb)

		sb = ams.calc_sb(amsdf)
		AMS = ams.AMS(sb[0],sb[1])
		if (VERBOSE): print 'AMS = {:<20}\nRough AMS = {:<20}\nDifference = {:<20}'.format(AMS,roughAMS, AMS - roughAMS)
		allams.append(AMS)

	avgAMS = np.array(allams).mean()
	avgrough = np.array(allrough).mean()
	if (VERBOSE) : print 'Averaged AMS = {}\nAveraged rough AMS = {}'.format(avgAMS, avgrough)
	outputline = '{ams:<15} {rams:<15} {model:<15} {datasize:<10} {nas:<15} {stand:<15} {folds:<10} {numf:<10} {params:<20}'.format(
				ams=avgAMS,
				rams=avgrough,
				model=MODEL,
				datasize=len(datadf),
				nas=NAS,
				stand=SCALE,
				folds=FOLDS,
				numf=len(FEATS),
				params=model_params)
	results[outputline] = avgAMS

"""	
sort report on mean AMS score
print to end of file (not worth rewriting file all over again to )
"""
if VERBOSE : print 'Preparing write out lines'
newout = ''
sorted_results = sorted(results.iteritems(),
						key=operator.itemgetter(1),
						reverse=True)
for string, ams in sorted_results :
	newout += '{out:<10}\n'.format(out=string)

newout += '\n'.join(FEATS)
newout += '\n\n**************************************************'

with open('models.log','r+') as log :
	content = log.read()
	log.seek(0,0)
	log.write(newout.rstrip('\r\n')+'\n'+content)



