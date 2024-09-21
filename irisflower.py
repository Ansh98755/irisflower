import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
st.write("""
##Simple Iris Flower Prediction App
# This App predicts the **Iris flower** type! """)

st.sidebar.header("User input parameters")

def user_input_features():
    sepal_length=st.sidebar.slider('sepal_length' , 4.3 , 7.9 , 5.4)
    sepal_width=st.sidebar.slider('sepal_width' , 2.0 , 4.4 , 3.4)
    petal_length=st.sidebar.slider('petal_length',1.0 , 6.9 , 1.3)
    petal_width=st.sidebar.slider('petal_width' , 0.1 , 2.5 , 0.2)
    data={'sepal_length':sepal_length,
          'sepal_width':sepal_width,
          'petal_length':petal_length,
          'petal_width':petal_width}
    features=pd.DataFrame(data,index=[0])
    return features
df=user_input_features()

st.subheader("User Input Parameters")
st.write(df)

iris=datasets.load_iris()

X=iris.data
Y=iris.target

Clf=RandomForestClassifier()
Clf.fit(X,Y)

Prediction= Clf.predict(df)
Prediction_proba=Clf.predict_proba(df)

st.subheader("Class labels and their corresponding index number")
st.write(iris.target_names)

st.subheader("Predictions")
st.write(iris.target_names[Prediction])

st.subheader("Prediction Probability")
st.write(Prediction_proba)