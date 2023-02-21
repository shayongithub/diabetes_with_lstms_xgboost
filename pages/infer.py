import streamlit as st 
import pandas as pd
import datetime
import pickle
import numpy as np 
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import scipy

st.markdown("## Inference Page 🔮")
st.sidebar.markdown("## Inference Page 🔮")

st.text('Input variables')
preg = st.number_input('Insert Pregnancies', min_value=0, key = 'preg', value = 7)
glucose = st.number_input('Insert Glucose', min_value=0, key = 'glucose', value = 187)
blood_pressure = st.number_input('Insert BloodPressure', min_value=0, key = 'blood_pressure', value = 68)
skin_thickness = st.number_input('Insert SkinThickness', min_value=0, key = 'skin_thickness', value = 39)
insulin = st.number_input('Insert Insulin', min_value=0, key = 'insulin', value = 304)
bmi = st.number_input('Insert BMI', min_value=0.0, key = 'bmi', value = 37.7)
diabetes_pedigree_function = st.number_input('Insert DiabetesPedigreeFunction', min_value=0.0, key = 'diabetes_function', value = 0.254)
age = int(st.number_input('Insert Age', min_value=0, max_value=125, key = 'age', value = 41))

model_name = st.selectbox('Pick model for inference', ['XGBoost_Normalized', 'XGBoost_Box', 'LSTMs_Normalized', 'LSTMs_Box_Cox'])

if model_name:

    threshold = float(st.text_input("Select prediction thresholds", 0.5))

    X = np.array([preg, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age])
    X = np.expand_dims(X, axis=0).reshape((1, 8))

    st.success(' Load input sucessfully', icon="✅")

    # ------- Load model -------
    if model_name.startswith('LSTMs'):

        model = keras.models.load_model(f'models/{model_name}.h5')
        
    else:
        
        with open(rf'models/{model_name}.pkl', 'rb') as pickle_file:
            model = pickle.load(pickle_file)
    # ------- Load model -------


    if st.button('Predict'):

        if model_name.endswith("Normalized"):
            scaler = MinMaxScaler(feature_range=(0, 1))
            normalized_X_test = scaler.fit_transform(X)

            if model_name.startswith('LSTMs'):  
                normalized_X_test = np.expand_dims(normalized_X_test.reshape(-1, 8), axis=0)
            
            y_pred = model.predict(normalized_X_test)
        else:
            col_names = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
            box_X_test = pd.DataFrame(X, columns=col_names)

            box_X_test = box_X_test.drop(["Insulin", "SkinThickness"], axis=1)

            # Add pseudo data as the scipy boxcox requrie at least 2 different columns and positive
            add_dict = {"Pregnancies": 1, "Glucose": 1, "BloodPressure": 1,
                        "BMI":1, "DiabetesPedigreeFunction":1,"Age":1}
            box_X_test = box_X_test.append(add_dict, ignore_index = True)
            
            # scipy return transformed data and the lambda used for tranforming
            box_X_test["Glucose"], fitted_lambda_glu = scipy.stats.boxcox(box_X_test["Glucose"],lmbda=None)
            box_X_test["BloodPressure"], fitted_lambda_blood_press = scipy.stats.boxcox(box_X_test["BloodPressure"],lmbda=None)
            box_X_test["BMI"], fitted_lambda_bmi = scipy.stats.boxcox(box_X_test["BMI"],lmbda=None)
            box_X_test["DiabetesPedigreeFunction"], fitted_lambda_dia_ped_func = scipy.stats.boxcox(box_X_test["DiabetesPedigreeFunction"],lmbda=None)
            box_X_test["Age"], fitted_lambda_age = scipy.stats.boxcox(box_X_test["Age"],lmbda=None)
            
            box_X_test = box_X_test.drop(labels=1, axis=0)

            
            if model_name.startswith('LSTMs'):  
                box_X_test = box_X_test.values.reshape((box_X_test.shape[0], 1, box_X_test.shape[1]))

            y_pred = model.predict(box_X_test)

        st.write(f'Prediction = {y_pred}')
        
        if isinstance(y_pred, float):
            y_pred = np.where(y_pred > threshold, 1, 0)

        if y_pred == 1:
            st.markdown('### The patient is likely to be *diabetic* 🐼') 
        else:
            st.markdown('### The patient is not likely to be *diabetic* 🐈')