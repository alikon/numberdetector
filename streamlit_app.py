import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


st.title("Perceptron")
df = pd.read_csv("dataset.csv")


x_train = df[["x1", "x2"]].values
y_train = df[["label"]].values

#
st.write(x_train.shape)
st.write(x_train)
st.write(y_train.shape)
st.write(y_train)
##st.write(np.bincount(y_train))
#
#plt.plot(
#    x_train[y_train == 0, 0],
#    x_train[y_train == 0, 1],
#    marker ="D",
#    markersize=13,
#    linestyle="",
#    label="Class 0",
#)
#
#plt.plot(
#    x_train[y_train == 1, 0],
#    x_train[y_train == 1, 1],
#    marker ="^",
#    markersize=13,
#    linestyle="",
#    label="Class 1",
#)
#
#plt.legend(loc=2)
#plt.xlim(-5, 5)
#plt.ylim(-5, 5)
#plt.xlabel("Feature $x_1$", fontsize=12)
#plt.ylabel("Feature $x_2$", fontsize=12)
#
#plt.grid()
#plt.show()
#
## X axis values:
x = [2,3,7,29,8,5,13,11,22,15]
# Y axis values:
y = [4,7,55,43,2,4,11,22,33,5]

fig = plt.figure()
# Create scatter plot:
plt.scatter(x, y)

st.pyplot(fig)