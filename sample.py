"""
    Sample usage of logistic regression model created in model_preparation.py.
    Uses Absenteeism module as a class
    """
from absenteeism_module import AbsenteeismModel


am = AbsenteeismModel()
am.load_clean_data('Absenteeism_new_data.csv')
am.describe_preprocessed_data()
am.describe_ready_data()

# logistic prediction of absenteeism basing on given data
dataWithResults = am.predict_absenteeism()
