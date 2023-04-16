import pandas as pd
import streamlit as st
import numpy as np
import matplotlib as mt


st.title("Perceptron")
df = pd.read_csv("dataset.csv")


x_train = df[["x1", "x2"]].values
y_train = df[["label"]]. values

#
st.write(x_train.shape())
st.write(x_train)
st.write(y_train.shape())
st.write(y_train)
st.write(np.bincount(y_train))