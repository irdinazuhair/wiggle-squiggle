import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

class HousingHomelessnessPrediction:
    def __init__(self, data_url):
        self.data_url = data_url
        self.df = None
        self.x_raw = None
        self.y_raw = None
        self.clf = RandomForestClassifier()

    def load_data(self):
        self.df = pd.read_csv(self.data_url)
        self.x_raw = self.df.drop('AtRiskOfOrExperiencingHomelessnessFlag', axis=1)
        self.y_raw = self.df['AtRiskOfOrExperiencingHomelessnessFlag']

    def preprocess_target(self):
        target_mapper = {'Yes': 1, 'No': 0}
        return self.y_raw.apply(lambda val: target_mapper[val])

    def encode_features(self, input_details):
        encode_cols = ['FamilyType', 'DisabilityApplicationFlag']
        encoded_df = pd.get_dummies(input_details, columns=encode_cols)
        return encoded_df

    def train_model(self, x, y):
        self.clf.fit(x, y)

    def predict(self, input_row):
        prediction = self.clf.predict(input_row)
        prediction_proba = self.clf.predict_proba(input_row)
        return prediction, prediction_proba

class StreamlitApp:
    def __init__(self, model):
        self.model = model

    def display_data(self):
        with st.expander('Data'):
            st.write('Dataset:')
            st.dataframe(self.model.df)
            st.write('**X (Features)**')
            st.dataframe(self.model.x_raw)
            st.write('**Y (Target)**')
            st.write(self.model.y_raw)

    def display_visualisation(self):
        with st.expander('Data Visualization'):
            st.scatter_chart(data=self.model.df, x='PeopleonApplication', y='FamilyType', color='AtRiskOfOrExperiencingHomelessnessFlag')

    def get_user_input(self):
        with st.sidebar:
            st.header('Input features')
            Family = st.selectbox('Family Type', (
                'Single Person', 
                'Single Parent, 1 Child', 
                'Single Parent, 2 Children', 
                'Single Parent, >2 Children', 
                'Couple Only', 
                'Couple, 1 Child', 
                'Couple, 2 Children', 
                'Couple, >2 Children', 
                'Single Person Over 55', 
                'Couple Only Over 55', 
                'Other'))
            DisabilityFlag = st.selectbox('Disability', ('Yes', 'No'))
            TotalPeople = st.slider('Total People on application', 1, 12, 2)
            TotalMonths = st.slider('Total months you have been register', 0, 239, 23)

            input_data = {
                'FamilyType': Family,
                'MonthsonHousingRegister': TotalMonths,
                'DisabilityApplicationFlag': DisabilityFlag,
                'PeopleonApplication': TotalPeople,
            }
            return pd.DataFrame(input_data, index=[0])

    def display_input(self, input_df, input_details):
        with st.expander('Input features'):
            st.write('**Input User**')
            st.dataframe(input_df)
            st.write('**Combined Housing data**')
            st.dataframe(input_details)

    def display_prepared_data(self, input_row, y):
        with st.expander('Data preparation'):
            st.write('**Encoded X (input housing)**')
            st.dataframe(input_row)
            st.write('**Encoded y**')
            st.write(y)

    def display_predictions(self, prediction, prediction_proba):
        housing_homeless = np.array(['Yes', 'No'])
        df_prediction_proba = pd.DataFrame(prediction_proba, columns=['No', 'Yes'])

        st.subheader('Predicted Homelessness')
        st.dataframe(
            df_prediction_proba,
            column_config={
                'Yes': st.column_config.ProgressColumn(
                    'Yes', format='%f', width='medium', min_value=0, max_value=1),
                'No': st.column_config.ProgressColumn(
                    'No', format='%f', width='medium', min_value=0, max_value=1),
            },
            hide_index=True
        )
        st.success(str(housing_homeless[prediction][0]))

# Main Execution
model = HousingHomelessnessPrediction("https://raw.githubusercontent.com/AleeyaHayfa/shiny-broccoli/refs/heads/master/Social_Housing_cleaned.csv")
model.load_data()
y_encoded = model.preprocess_target()

app = StreamlitApp(model)

app.display_data()
app.display_visualisation()

input_df = app.get_user_input()
input_details = pd.concat([input_df, model.x_raw], axis=0)

app.display_input(input_df, input_details)

df_house = model.encode_features(input_details)
x_encoded = df_house[1:]
input_row_encoded = df_house[:1]

app.display_prepared_data(input_row_encoded, y_encoded)

model.train_model(x_encoded, y_encoded)

prediction, prediction_proba = model.predict(input_row_encoded)
app.display_predictions(prediction, prediction_proba)
