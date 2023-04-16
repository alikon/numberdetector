import pandas as pd
import streamlit as st
import numpy as np
import matplotlib as plt


st.title("Perceptron")
df = pd.read_csv("dataset.csv")


x_train = df[["x1", "x2"]].values
y_train = df[["label"]].values

#
st.write(x_train.shape)
st.write(x_train)
st.write(y_train.shape)
st.write(y_train)
#st.write(np.bincount(y_train))

plt.plot(
    x_train[y_train == 0, 0],
    x_train[y_train == 0, 1],
    marker ="D",
    markersize=13,
    linestyle="",
    label="Class 0"
)

plt.plot(
    x_train[y_train == 1, 0],
    x_train[y_train == 1, 1],
    marker ="^",
    markersize=13,
    linestyle="",
    label="Class 1"
)

plt.legend(loc=2)
plt.xlimit("Feature $x_1$", fontsize=12)
plt.ylimit("Feature $x_2$", fontsize=12)

plt.grid()
plt.show()