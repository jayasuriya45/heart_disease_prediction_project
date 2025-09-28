import pickle
import streamlit as st
import numpy as np

with open (r"xgb_model.pkl","rb") as pickle_file:
    loaded_model=pickle.load(pickle_file)
    
    
    

res=loaded_model.predict([[1	,1,128,	2,	0,	0,	2	,308,	0.0	,1	,45,	170,	0]])    
 
if res[0] == 0:
    print("No heart disease")
else:
    print("Heart disease detected")


st.title(" Heart Disease Predictor")

st.write("Enter the patient details to predict heart disease:")

# Create input fields for each feature
# Adjust the number of inputs based on your model's expected features
slope = st.number_input("Slope of Peak Exercise ST Segment", value=2)
thal = st.number_input("Thalassemia (3=Normal, 6=Fixed Defect, 7=Reversible Defect)", value=0)
trestbps = st.number_input("Resting Blood Pressure", value=128)
cp = st.number_input("Chest Pain Type", value=1)
ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", value=0)
fbs = st.number_input("Fasting Blood Sugar >120 mg/dl (1=True, 0=False)", value=0)
restecg = st.number_input("Resting EKG Results", value=2)
chol = st.number_input("Serum Cholesterol (mg/dl)", value=170)
oldpeak = st.number_input("ST Depression Induced by Exercise", value=0.0)
sex = st.number_input("Sex (1=Male, 0=Female)", value=1)
age = st.number_input("Age", value=45)
thalach = st.number_input("Max Heart Rate Achieved", value=308)
exang = st.number_input("Exercise Induced Angina (1=Yes, 0=No)", value=0)

# When the user clicks the Predict button
if st.button("Predict"):
    features = np.array([[slope, thal, trestbps, cp, ca, fbs, restecg,
                          chol, oldpeak, sex, age, thalach, exang]])
    
    res = loaded_model.predict(features)
    
    if res[0] == 0:
        st.success("No heart disease detected")
    else:
        st.error(" Heart disease detected")   