# AbsenteesimRegressionModel
 A case study with logistic regression model for absence in work basing on data such as:
 travel costs, day of the week,  distance to work, and more.
 Code based on Udemy course with some modifications

Absenteeism_data.csv - data for model training and testing
Absenteeism_new_data.csv - input data for sample.py 
Absenteeism_preprocessed.csv - preprocessed data by datapreprocessing.py, removed irrelevant columns, dealing with dummies and dates
datapreprocessing.py - creates Absenteeism_preprocessed.csv
model_preparation.py - logistic regression model, exported to model_Absenteesim and model_Absenteesim_scaler 
absenteeism_module.py - contains class AbsenteeismModel and methods to load and clean the new dataset to use exported models 
sample.py - a sample usage of class AbsenteeismModel on a new Absenteeism_new_data.csv - input data for sample.py 


