# Machine Learning Tools
A couple scripts I wrote when I was working on a machine learning challenge on Kaggle. 

`daily_log.py` provides a means to periodically write update notes on the project your working on. It prompts you for a message as to what you've done since you've last written, and then writes it out, along with the files that have been modified in its directory to the `daily.log` file.

`one_model_param.py` and `one_model_feat.py` take in a set of parameters for a given model or a set of features from a given data set, and train all versions of the model with combinations those parameter settings (or features). Results are written out to `models.log`
