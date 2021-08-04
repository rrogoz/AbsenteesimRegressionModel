import pandas as pd
import numpy as np

raw_csv_data = pd.read_csv('Absenteeism_data.csv')
df = raw_csv_data.copy()

# removing irrelevant columns: ID
df = df.drop(['ID'], axis=1)

# create 0,1 from Reasons dummies
reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first=True)

# grouping the reasons
reasons_classified = pd.DataFrame()
reasons_classified['reason_type_1'] = reason_columns.loc[:, 1:14].max(axis=1)
reasons_classified['reason_type_2'] = reason_columns.loc[:, 15:17].max(axis=1)
reasons_classified['reason_type_3'] = reason_columns.loc[:, 18:21].max(axis=1)
reasons_classified['reason_type_4'] = reason_columns.loc[:, 22:].max(axis=1)
df = df.drop(['Reason for Absence'], axis=1)
df = pd.concat([df, reasons_classified], axis=1)

# reordering columns
cols = df.columns.tolist()
reason1_index = cols.index('reason_type_1')
cols = cols[reason1_index:] + cols[:reason1_index]
df = df[cols]

# dealing with Date column
df_reason_mod = df.copy()
df_reason_mod['Date'] = pd.to_datetime(
    df_reason_mod['Date'], format='%d/%m/%Y')

month_list = []
weekday_list = []
for i in range(df_reason_mod['Date'].shape[0]):
    month_list.append(df_reason_mod['Date'][i].month)
    weekday_list.append(df_reason_mod['Date'][i].weekday())

df_reason_mod['Month'] = month_list
df_reason_mod['Day of the week'] = weekday_list

df_reason_and_data_mod = df_reason_mod.drop(['Date'], axis=1)

# education mapping from 4 educations degress create (0,1)
df_reason_and_data_mod['Education'] = df_reason_and_data_mod['Education'].map({
                                                                              1: 0, 2: 1, 3: 1, 4: 1}
                                                                              )
# reordering columns
cols = df_reason_and_data_mod.columns.tolist()
target_index = cols.index('Absenteeism Time in Hours')
cols = cols[:target_index] + cols[target_index + 1:] + \
    cols[target_index:target_index+1]
df_reason_and_data_mod = df_reason_and_data_mod[cols]
#
df_preprocessed = df_reason_and_data_mod.copy()
df_preprocessed.to_csv('Absenteeism_preprocessed.csv', index=False)
