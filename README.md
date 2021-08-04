# AbsenteesimRegressionModel
 A case study with logistic regression model

Absenteeism_data.csv - data for model training and testing
Absenteeism_new_data.csv - input data for sample.py 
Absenteeism_preprocessed.csv - preprocessed data by datapreprocessing.py, removed irrelevant columns, dealing with dummies and dates
datapreprocessing.py - creates data above
model_preparation.py - logistic regression model, exported to model_Absenteesim and model_Absenteesim_scaler 
absenteeism_module.py - contains class AbsenteeismModel and methods to load and clean the new dataset to use exported models 
sample.py - a sample usage of class AbsenteeismModel on a new Absenteeism_new_data.csv - input data for sample.py 


