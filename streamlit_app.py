import pandas as pd
import streamlit as st


st.title("Perceptron")
df = pd.read_csv("dataset.csv"header=0,
        usecols=["x1", "x2", "label"])
st.write(df.to_string())

#x_train = df[["x1", "x2"]].values
#y_train = df[["label"]]. values
#
#st.write(x_train)
#st.write(y_train)