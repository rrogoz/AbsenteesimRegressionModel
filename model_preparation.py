import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression
import pickle

# reading the data
data_preprocessed = pd.read_csv('Absenteeism_preprocessed.csv')

# simple assumptions as follow:
# absent > median -> 1
# absent < median -> 0

median = data_preprocessed['Absenteeism Time in Hours'].median()
targets = np.where(
    data_preprocessed['Absenteeism Time in Hours'] > median, 1, 0)

data_with_targets = data_preprocessed.drop(
    ['Absenteeism Time in Hours'], axis=1)
data_with_targets['Excessive Absenteeism'] = targets

# dividing to inputs and targets
data_input_unscaled = data_with_targets.iloc[:, :-1]
data_targets = data_with_targets.iloc[:, -1]

# dividing into dummies and rest to avoid dummies standardization
cols_to_not_scale = ['reason_type_1', 'reason_type_2',
                     'reason_type_3', 'reason_type_4', 'Education']
data_input_to_scale = data_input_unscaled.drop(cols_to_not_scale, axis=1)

# standarazing the data
scaler = StandardScaler()
scaler.fit(data_input_to_scale)
data_input_scaled_ndarray = scaler.transform(data_input_to_scale)
data_input_scaled = pd.DataFrame(
    data=data_input_scaled_ndarray, columns=data_input_to_scale.columns.values)

# replacing non-standardized data with scaled ones
ready_input_data = data_input_unscaled.copy()
for column in data_input_scaled:
    ready_input_data[column] = data_input_scaled[column]

# split the data into train and test
x_train, x_test = train_test_split(
    ready_input_data, test_size=0.25, random_state=25)
y_train, y_test = train_test_split(
    data_targets, test_size=0.25, random_state=25)

# create logistic regression
reg = LogisticRegression()
reg.fit(x_train, y_train)
y_predicted = reg.predict(x_train)

R = reg.score(x_train, y_train)
featuresSelection = f_regression(x_train, y_train)
# create summary table
reg_sum = pd.DataFrame(
    data=data_input_unscaled.columns, columns=['Parameter'])
reg_sum['Weights'] = reg.coef_.T
reg_sum['p-values'] = featuresSelection[1]
reg_sum.index = reg_sum.index + 1
reg_sum.loc[0] = ['Intercept', reg.intercept_[0], 0.]
reg_sum = reg_sum.sort_index()
# note! those close to 1 are irrelevant for model
reg_sum['Odds Ratio'] = np.exp(reg_sum['Weights'])

# Backward data selections
# removing inputs with p>0.05 

cols_to_remove = reg_sum[reg_sum['p-values'] > 0.05]
# cols_to_remove = cols_to_remove.loc[
#     ~(cols_to_remove == 'reason_type_2').max(axis=1)]  # keep all reasons

ready_input_data_reduced = ready_input_data.drop(
    cols_to_remove['Parameter'].tolist(), axis=1)

# create logistic regression for reduced model
x_train_reduced, x_test_reduced = train_test_split(
    ready_input_data_reduced, test_size=0.25, random_state=25)

reg_reduced = LogisticRegression()
reg_reduced.fit(x_train_reduced, y_train)
y_predicted_reduced = reg_reduced.predict(x_train_reduced)

R_reduced = reg_reduced.score(x_train_reduced, y_train)
featuresSelection_reduced = f_regression(x_train_reduced, y_train)
# create summary table for reduced model
reg_sum_reduced = pd.DataFrame(
    data=ready_input_data_reduced.columns, columns=['Parameter'])
reg_sum_reduced['Weights'] = reg_reduced.coef_.T
reg_sum_reduced['p-values'] = featuresSelection_reduced[1]
reg_sum_reduced.index = reg_sum_reduced.index + 1
reg_sum_reduced.loc[0] = ['Intercept', reg_reduced.intercept_[0], 0.]
reg_sum_reduced = reg_sum_reduced.sort_index()
reg_sum_reduced['Odds Ratio'] = np.exp(reg_sum_reduced['Weights'])


# Test the model
R_test_reduced = reg_reduced.score(x_test_reduced, y_test)
predicted_proba = reg_reduced.predict_proba(x_test_reduced)

# save the model
with open('model_Absenteeism', 'wb') as file:
    pickle.dump(reg_reduced, file)
with open('model_Absenteeism_scaler', 'wb') as file1:
    pickle.dump(scaler, file1)
