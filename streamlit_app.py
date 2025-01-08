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
    X_raw = df.drop('AtRiskOfOrExperiencingHomelessnessFlag', axis=1)
    st.dataframe(X_raw)

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
  TotalPeople = st.slider('PeopleonApplication', 1.0, 12.0, 1.681)
  TotalMonths = st.slider('MonthsonHousingRegister', 0.0, 239.0, 23.3)
  DisabilityFlag = st.selectbox('DisabilityApplicationFlag', ('Yes', 'No'))
  # 32.1 mininum, 59.6 maximum, 43.9 average (same as other)

#  # user input into dataframe
# # Create a DataFrame for the input features
#   data = {'island': island,
#           'bill_length_mm': bill_length_mm,
#           'bill_depth_mm': bill_depth_mm,
#           'flipper_length_mm': flipper_length_mm,
#           'body_mass_g': body_mass_g,
#           'sex': gender}
#   input_df = pd.DataFrame(data, index=[0])
#   input_penguins = pd.concat([input_df, x_raw], axis=0) #combine input features with penguin features
