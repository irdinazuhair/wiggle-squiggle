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
  st.scatter_chart(data=df, x='PeopleonApplication', y='ApplicationType', color='AtRiskOfOrExperiencingHomelessnessFlag')
