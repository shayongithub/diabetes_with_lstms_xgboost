import streamlit as st 
import pandas as pd
import scipy
import os
from pathlib import Path
import numpy as np

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

import pickle

st.markdown("# Compare models 💽")
st.sidebar.markdown("## Compare models 💽")

# Load test data
st.markdown("### Load Test Data (in .csv format)")
st.warning("The test data should be in the same format as the `diabetes.csv` file")

def load_model_by_path(model_path):

    with open(model_path, "rb") as pickle_file:
        model = pickle.load(pickle_file)

    return model

#  ------------ Upload Testdata ------------
uploaded_file = st.file_uploader("Choose a CSV file", type={"csv"})

if uploaded_file:
    test_df = pd.read_csv(uploaded_file)

    # Extract the test data 
    X_test = test_df.drop(["Outcome"], axis=1)
    y_test = test_df['Outcome']

    #  ------------ Upload Testdata ------------


    #  ------------ Select models ------------
    if st.button("Show the first 10 rows of test data"):
        st.write(test_df.head(10))

    model_type = ["XGBoost", "LSTMs"]

    model = st.selectbox("Pick your model for metrics on datatest2", ["Compare all models"] + model_type)
    #  ------------ Select models ------------


    if model != "Compare all models":
        ml_type = st.selectbox("Pick your model for metrics on datatest2", ["Normalization", "Box-Cox Transformation"])

        threshold = float(st.text_input("Select prediction thresholds", 0.5))
        
        if ml_type == "Normalization":
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            process_X_test = scaler.fit_transform(X_test)

            if model == "XGBoost":
                
                load_model = load_model_by_path("models/XGBoost_Normalized.pkl")

                if load_model:
                    st.success(' Load trained model succesfully', icon="✅")

            else:
                load_model = keras.models.load_model("models/LSTMs_Normalized.h5")

                if load_model:
                    st.success(' Load trained model succesfully', icon="✅")

        elif ml_type == "Box-Cox Transformation":

            process_X_test = X_test.copy()
            process_X_test = process_X_test.drop(["Insulin", "SkinThickness"], axis=1)
            # scipy return transformed data and the lambda used for tranforming
            process_X_test["Glucose"], fitted_lambda_glu = scipy.stats.boxcox(process_X_test["Glucose"],lmbda=None)
            process_X_test["BloodPressure"], fitted_lambda_blood_press = scipy.stats.boxcox(process_X_test["BloodPressure"],lmbda=None)
            process_X_test["BMI"], fitted_lambda_bmi = scipy.stats.boxcox(process_X_test["BMI"],lmbda=None)
            process_X_test["DiabetesPedigreeFunction"], fitted_lambda_dia_ped_func = scipy.stats.boxcox(process_X_test["DiabetesPedigreeFunction"],lmbda=None)
            process_X_test["Age"], fitted_lambda_age = scipy.stats.boxcox(process_X_test["Age"],lmbda=None)
            
            process_X_test = process_X_test.values
            if model == "XGBoost":

                load_model = load_model_by_path("models/XGBoost_Box.pkl")

                if load_model:
                    st.success(' Load trained model succesfully', icon="✅")

            else:
                load_model = keras.models.load_model("models/LSTMs_Box_Cox.h5")

                if load_model:
                    st.success(' Load trained model succesfully', icon="✅")


        if model == "LSTMs":
            process_X_test = process_X_test.reshape((process_X_test.shape[0], 1, process_X_test.shape[1]))
            y_pred = load_model.predict(process_X_test)
            roc_auc_score = roc_auc_score(y_test, y_pred)
        else:
            y_pred = load_model.predict_proba(process_X_test)[:,1]
            roc_auc_score = roc_auc_score(y_test, y_pred)

        st.text("ROC-AUC Score:")
        st.write(roc_auc_score)
        st.text("Classification Report:")
        y_pred_clf_rp = np.where(y_pred > threshold, 1, 0)
        clf_report = classification_report(y_test, y_pred_clf_rp, output_dict = True)
        df_classification_report = pd.DataFrame(clf_report).transpose()
        # df_classification_report = df_classification_report.sort_values(by=["f1-score"], ascending=False)
        st.table(df_classification_report)

    elif model == "Compare all models":

        if st.button('Show results on Test data'):

            st.info("Running test data on 3 models")
            # Load 3 model
            results = {} # {name: [] for name in all_model}

            all_model = [Path(os.path.join("models", file)) for file in os.listdir(r"models") if os.path.isfile(os.path.join("models", file))]

            # ---- Normalization ----
            scaler_all_model = MinMaxScaler(feature_range=(0, 1))
            normalized_X_test = scaler_all_model.fit_transform(X_test)
            # ---- Normalization ----

            # ---- Box-Cox ----
            box_X_test = X_test.copy()
            box_X_test = box_X_test.drop(["Insulin", "SkinThickness"], axis=1)
            # scipy return transformed data and the lambda used for tranforming
            box_X_test["Glucose"], fitted_lambda_glu = scipy.stats.boxcox(box_X_test["Glucose"],lmbda=None)
            box_X_test["BloodPressure"], fitted_lambda_blood_press = scipy.stats.boxcox(box_X_test["BloodPressure"],lmbda=None)
            box_X_test["BMI"], fitted_lambda_bmi = scipy.stats.boxcox(box_X_test["BMI"],lmbda=None)
            box_X_test["DiabetesPedigreeFunction"], fitted_lambda_dia_ped_func = scipy.stats.boxcox(box_X_test["DiabetesPedigreeFunction"],lmbda=None)
            box_X_test["Age"], fitted_lambda_age = scipy.stats.boxcox(box_X_test["Age"],lmbda=None)
            # ---- Box-Cox ----

            # ---- LSTM ---
            lstm_normalized_X_test = normalized_X_test.reshape((normalized_X_test.shape[0], 1, normalized_X_test.shape[1]))
            lstm_box_X_test = box_X_test.values.reshape((box_X_test.shape[0], 1, box_X_test.shape[1]))
            # ---- LSTM ---

            for model_path in all_model:
                
                st.write(f"Model path: {str(model_path.name)}")

                if model_path.name.startswith('lstms_model'):
                    model = keras.models.load_model(model_path)
                else:
                    model = load_model_by_path(model_path)             
                
                if model_path.name == "XGBoost_Normalized.pkl":

                    # y_pred_xgb_scaled = model.predict(normalized_X_test)
                    y_pred_xgb_scaled = model.predict_proba(normalized_X_test)[:,1]
                    results["XGBoost_Normalized"] = roc_auc_score(y_test, y_pred_xgb_scaled)

                elif model_path.name == "XGBoost_Box.pkl":

                    # y_pred_xgb_box = model.predict(box_X_test)
                    y_pred_xgb_box = model.predict_proba(box_X_test)[:,1]
                    results["XGBoost_Box-Cox"] = roc_auc_score(y_test, y_pred_xgb_box)

                elif model_path.name == "LSTMs_Normalized.h5":
                    y_pred_lstm_scaled = model.predict(lstm_normalized_X_test)
                    # y_pred_lstm_scaled = np.where(y_pred_lstm_scaled > threshold, 1, 0)
                    results["LSTMs_Normalized"] = roc_auc_score(y_test, y_pred_lstm_scaled)
                
                else:
                    y_pred_box_cox = model.predict(lstm_box_X_test)
                    # y_pred_box_cox = np.where(y_pred_box_cox > threshold, 1, 0)
                    results["LSTMs_Box_Cox"] = roc_auc_score(y_test, y_pred_box_cox)


            res_df = pd.DataFrame.from_dict(results, orient = "index", columns=["ROC-AUC scores"])
            st.write(res_df)