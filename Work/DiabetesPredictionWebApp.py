# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 21:21:40 2025

@author: Suprava Modak
"""

import numpy as np
import streamlit as st
import pickle

loaded_model=pickle.load(open('trained_model.sav','rb'))


def diabetes_prediction(input_data):
    #change the input_data to numpy array
    input_data_as_numpy_array=np.asarray(input_data)

    #reshape the array as we are predicting for one instance
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)

    if(prediction[0]==1):
            return "The person is Diabetic"
    else:
            return"The person is not Diabetic"


def main():
    st.title('Diabetes Prediction Web App')
    
    Pregnancies=st.text_input('Number of Pregnancies')
    Glucose=st.text_input('Glucose Level')
    BloodPressure=st.text_input('Blood Pressure value')
    SkinThickness=st.text_input('Skin Thickness value')
    Insulin=st.text_input('Insulin Level')
    BMI=st.text_input('BMI value')
    DiabetesPedigreeFunction=st.text_input('Diabetes Pedigree Fuction value')
    Age=st.text_input('Age of Person')
    
    
    diagnosis=''
    
    if st.button('Diabetes Test Result'):
        diagnosis=diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,BMI, DiabetesPedigreeFunction, Age])
        
    st.success(diagnosis)  
    
if __name__ =='__main__':
    main()
