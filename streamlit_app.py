import streamlit as st
import pandas as pd

st.title('🎈NexusGo')

st.write('This is an app testing for Assignment CPT316!')

with st.expander('Data'):
  df = pd.read_csv("https://raw.githubusercontent.com/AleeyaHayfa/shiny-broccoli/refs/heads/master/Social_Housing_cleaned.csv")
  df

  st.write('**X**')
  x_raw = df.drop('Diabetes_binary', axis=1)
  x_raw

  st.write('Y')
  y_raw = df.Diabetes_binary
  y_raw

with st.expander('Data Visualization'):
  st.scatter_chart(data=df, x='HighBp', y='BMI', color='Diabetes_binary')
