import streamlit as st
import pandas as pd

# Header for the main Content
st.header('Two Stage Operational Amplifier')

# Header for the sidebar
st.sidebar.header('User Input Features')

df = pd.read_csv('2STAGEOPAMP_DATASET.csv')
st.subheader('Description of the Dataset')
columns = ['Is4','Gm6','Gm4','Asp_1','Asp_2','Asp_3','Asp_4','Asp_5','Abs_Gain','Delay']
df.drop(columns,axis='columns',inplace=True)

st.write(df.describe())
