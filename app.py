import numpy as np
import pickle
import streamlit as st

loaded_model=pickle.load(open('trained_model.sav','rb'))

def diabetic_prediction(input_data):
 
    input_data=np.asarray(input_data)
    input_data_reshaped=input_data.reshape(1,-1)

    prediction=loaded_model.predict(input_data_reshaped)

    print(prediction)
    if(prediction==0):
       return 'the person is not diabetic'
    else:
       return 'the person is  diabetic'  

def main():
    st.title("Diabetes Prediction Web app ")

    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')

    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetic_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
        
    st.success(diagnosis)
    
if __name__ == '__main__':
    main()   