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

    # Feature selection using SelectKBest
    k = 10  # Choose the top k features
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X_raw, y_raw)

    # Display the selected features and their scores
    selected_features = X_raw.columns[selector.get_support()]
    scores = selector.scores_

    st.write('**Selected Features:**', list(selected_features))
    st.write('**Feature Scores:**')
    feature_scores = pd.DataFrame({'Feature': X_raw.columns, 'Score': scores})
    st.dataframe(feature_scores.sort_values(by='Score', ascending=False))

with st.expander('Data Visualization'):
  st.scatter_chart(data=df, x='PeopleonApplication', y='ApplicationType', color='AtRiskOfOrExperiencingHomelessnessFlag')
