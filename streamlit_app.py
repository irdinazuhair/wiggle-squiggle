import streamlit as st
import pandas as pd

st.title('🌱ThinkTankers Model')

st.write('This is model developement for Assignment CPT316!')

with st.expander('Data'):
  df = pd.read_csv("https://raw.githubusercontent.com/AleeyaHayfa/shiny-broccoli/refs/heads/master/Social_Housing_cleaned.csv")
  df

  st.write('**X**')
  x_raw = df.drop('AtRiskOfOrExperiencingHomelessnessFlag', axis=1)
  x_raw

  st.write('Y')
  y_raw = df.AtRiskOfOrExperiencingHomelessnessFlag
  y_raw
