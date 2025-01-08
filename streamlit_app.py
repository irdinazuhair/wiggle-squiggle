from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
import streamlit as st

with st.expander('Data'):
    # Load the dataset
    df = pd.read_csv("https://raw.githubusercontent.com/AleeyaHayfa/shiny-broccoli/refs/heads/master/Social_Housing_cleaned.csv")
    st.write('Dataset:')
    st.dataframe(df)

    # Separate features (X) and target (Y)
    st.write('**X (Features)**')
    x_raw = df.drop('AtRiskOfOrExperiencingHomelessnessFlag', axis=1)
    st.dataframe(x_raw)

    st.write('**Y (Target)**')
    y_raw = df['AtRiskOfOrExperiencingHomelessnessFlag']
    st.write(y_raw)

with st.expander('Data Visualization'):
  st.scatter_chart(data=df, x='PeopleonApplication', y='FamilyType', color='AtRiskOfOrExperiencingHomelessnessFlag')

# Data prepration (user input)
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
    'Other'
))
  DisabilityFlag = st.selectbox('Disability', ('Yes', 'No'))
  TotalPeople = st.slider('Total People on application', 1.0, 12.0, 1.681)
  TotalMonths = st.slider('Total months you have been register', 0.0, 239.0, 23.3)
  # 1.0 mininum, 12.0 maximum, 1.681 average (same as other)

# user input into dataframe
# Create a DataFrame for the input features
  data = {'FamilyType': Family,
          'MonthsonHousingRegister': TotalMonths,
          'DisabilityApplicationFlag': DisabilityFlag,
          'PeopleonApplication': TotalPeople,
         }
  input_df = pd.DataFrame(data, index=[0])
  input_details = pd.concat([input_df, x_raw], axis=0) #combine input features with penguin features

with st.expander('Input features'):
  st.write('**Input User**')
  input_df
  st.write('**Combined Housing data**') #combine original dgn user input
  input_details
