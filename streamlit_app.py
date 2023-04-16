import pandas as pd
import streamlit as st


st.title("Perceptron")
df = pd.read_csv("dataset.csv")

x_train = df[["x1", "x2"]].values
y_train = df[["label"]]. values

st.write(x_train)
st.write(y_train)