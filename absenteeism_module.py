import pandas as pd
import pickle


class AbsenteeismModel:
    def __init__(self,
                 model_file='model_Absenteeism',
                 scaler_file='model_Absenteeism_scaler') -> None:
        # import logistic regression model
        with open(model_file, 'rb') as modelFile:
            # import Scaler
            self.reg = pickle.load(modelFile)
        with open(scaler_file, 'rb') as scalerFile:
            self.scaler = pickle.load(scalerFile)
        self.input_data = pd.DataFrame()
        self.ready_data = pd.DataFrame()
        self.preprocessed_data = pd.DataFrame()

    def load_clean_data(self, dataFile):
        """ Load, clean and scale the data in order that is required
         by regression model
        Args:
            dataFile ([str]): [csv file]
        """
        self.input_data = pd.read_csv(dataFile)
        df = self.input_data.copy()

        # removing irrelevant columns: ID
        df = df.drop(['ID'], axis=1)

        # create 0,1 from Reasons dummies
        reason_columns = pd.get_dummies(
            df['Reason for Absence'], drop_first=True)

        # grouping the reasons
        reasons_classified = pd.DataFrame()
        reasons_classified['reason_type_1'] = reason_columns.loc[:, 1:14].max(
            axis=1)
        reasons_classified['reason_type_2'] = reason_columns.loc[:, 15:17].max(
            axis=1)
        reasons_classified['reason_type_3'] = reason_columns.loc[:, 18:21].max(
            axis=1)
        reasons_classified['reason_type_4'] = reason_columns.loc[:, 22:].max(
            axis=1)
        df = df.drop(['Reason for Absence'], axis=1)
        df = pd.concat([df, reasons_classified], axis=1)

        # reordering columns
        cols = df.columns.tolist()
        reason1_index = cols.index('reason_type_1')
        cols = cols[reason1_index:] + cols[:reason1_index]
        df = df[cols]

        # dealing with Date column
        df['Date'] = pd.to_datetime(
            df['Date'], format='%d/%m/%Y')

        month_list = []
        weekday_list = []
        for i in range(df['Date'].shape[0]):
            month_list.append(df['Date'][i].month)
            weekday_list.append(df['Date'][i].weekday())

        df['Month'] = month_list
        df['Day of the week'] = weekday_list
        df = df.drop(['Date'], axis=1)

        # education mapping
        df['Education'] = df['Education'].map({
            1: 0, 2: 1, 3: 1, 4: 1}
        )
        # removing columns which are not need for model
        cols_to_remove = ['reason_type_2',
                          'Distance to Work',
                          'Age',
                          'Daily Work Load Average',
                          'Body Mass Index',
                          'Education',
                          'Pets',
                          'Month']
        df = df.drop(cols_to_remove, axis=1)
        # dividing into dummies and rest to avoid dummies standardization
        cols_to_not_scale = ['reason_type_1',
                             'reason_type_3', 'reason_type_4']
        data_input_to_scale = df.drop(
            cols_to_not_scale, axis=1)
        self.preprocessed_data = df.copy()
        # standarazing the data
        self.scaler.fit(data_input_to_scale)
        data_input_scaled_ndarray = self.scaler.transform(data_input_to_scale)
        data_input_scaled = pd.DataFrame(
            data=data_input_scaled_ndarray, columns=data_input_to_scale.columns.values)
        ready_input_data = df.copy()
        for column in data_input_scaled:
            ready_input_data[column] = data_input_scaled[column]
        self.ready_data = ready_input_data.copy()

    def describe_preprocessed_data(self):
        print(self.preprocessed_data.describe())

    def describe_ready_data(self):
        print(self.ready_data.describe())

    def predict_absenteeism(self):
        predicts = self.reg.predict(self.ready_data)
        predicts_proba = self.reg.predict_proba(self.ready_data)
        frameWithPredictions = self.input_data.copy()
        frameWithPredictions['Predictions of absenteeism'] = predicts
        frameWithPredictions['Probability of absenteeism'] = predicts_proba[:, 1]
        return frameWithPredictions
