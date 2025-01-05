import streamlit as st
import pandas as pd

st.title('🎈JomPredict')

st.write('This is an app testing for Assignment CPT316!')

df = pd.read_csv("https://raw.githubusercontent.com/AleeyaHayfa/shiny-broccoli/refs/heads/master/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
df
