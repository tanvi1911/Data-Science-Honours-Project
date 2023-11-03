# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tempfile import NamedTemporaryFile


df = pd.read_csv('HealthData.csv')
df = df.rename(columns={'num':'target'})

df['ca'] = df['ca'].fillna(df['ca'].mean())
df['thal'] = df['thal'].fillna(df['thal'].mean())

X = df.drop(columns = 'target')
y = df.target

# splitting our dataset into training and testing for this we will use train_test_split library.
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.25, random_state=42)
print('X_train size: {}, X_test size: {}'.format(X_train.shape, X_test.shape))

from sklearn.preprocessing import StandardScaler

#feature scaling
scaler= StandardScaler()
X_train_scaler= scaler.fit_transform(X_train)
X_test_scaler= scaler.fit_transform(X_test)


# Create and train the model
model= LogisticRegression()
model.fit(X_train_scaler, y_train)
# Load your trained machine learning model
# model_path = r'D:\Tanvi2022\Data Science Honours Project\naive_bayes_model.pkl'
# joblib.load(model_path)



# Define explanations for medical terms
term_explanations = {
    'Age': "Age of the patient.",
    'Gender': "Gender of the patient.",
    'cp': "Type of chest pain experienced by the patient. It can have four values.",
    'trestbps': "The resting blood pressure of the patient in mm Hg.",
    'chol': "Cholesterol levels in mg/dl.",
    'fbs': "Whether the fasting blood sugar is greater than 120 mg/dl (0 for No, 1 for Yes).",
    'restecg': "Results of resting electrocardiogram. It can have three values.",
    'thalach': "The maximum heart rate achieved during exercise.",
    'exang': "Whether exercise induced angina is present (0 for No, 1 for Yes).",
    'oldpeak': "ST depression induced by exercise relative to rest.",
    'slope': "The slope of the peak exercise ST segment. It can have three values.",
    'ca': "The number of major blood vessels in the heart that were visualized and appear to have some blood flow.",
    'thal': "The result of the thallium stress test. It can have three values.",
}


# Create the Streamlit web app
st.title('Heart Disease Prediction')
selected_term = st.selectbox('Select a Medical Term', list(term_explanations.keys()))


# Display the explanation based on the selected term
if selected_term in term_explanations:
    explanation = term_explanations[selected_term]
    st.write(f'**{selected_term.capitalize()} Explanation:** {explanation}')

# Add input fields for user data
st.write('Please enter the following information:')
age = st.slider('Age', 20, 100, 50)
sex = st.radio('Gender', ['Male', 'Female'])
cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'])
trestbps = st.slider('Resting Blood Pressure (mm Hg)', 90, 200, 120)
chol = st.slider('Cholesterol (mg/dl)', 100, 400, 200)
fbs = st.radio('Fasting Blood Sugar > 120 mg/dl', ['Yes', 'No'])
restecg = st.selectbox('Resting Electrocardiographic Results', ['Normal', 'ST-T wave abnormality', 'Probable or definite left ventricular hypertrophy'])
thalach = st.slider('Maximum Heart Rate Achieved', 70, 200, 150)
exang = st.radio('Exercise Induced Angina', ['Yes', 'No'])
oldpeak = st.slider('ST Depression Induced by Exercise Relative to Rest', 0.0, 6.2, 1.0)
slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])
ca = st.slider('Number of Major Vessels Colored by Flouroscopy', 0, 3, 0)
thal = st.selectbox('Thallium Stress Test Result', ['Normal', 'Fixed Defect', 'Reversible Defect'])

# Convert user input to numerical values
sex = 1 if sex == 'Male' else 0
fbs = 1 if fbs == 'Yes' else 0
exang = 1 if exang == 'Yes' else 0

# Map categorical values to numerical values
cp_dict = {
    'Typical Angina': 0,
    'Atypical Angina': 1,
    'Non-Anginal Pain': 2,
    'Asymptomatic': 3
}
restecg_dict = {
    'Normal': 0,
    'ST-T wave abnormality': 1,
    'Probable or definite left ventricular hypertrophy': 2
}
slope_dict = {
    'Upsloping': 0,
    'Flat': 1,
    'Downsloping': 2
}
thal_dict = {
    'Normal': 0,
    'Fixed Defect': 1,
    'Reversible Defect': 2
}

model_path = r'D:\Tanvi2022\Data Science Honours Project\naive_bayes_model.pkl'
model = joblib.load(model_path)

# # Load the model from a file
# with open('naive_bayes_model.pkl', 'rb') as file:
#     loaded_model = pickle.load(file)

# Make predictions
# data = np.array([age, sex, cp_dict[cp], trestbps, chol, fbs, restecg_dict[restecg], thalach, exang, oldpeak, slope_dict[slope], ca, thal_dict[thal]]).reshape(1, -1)

# prediction = model.predict(data)

# # Display prediction
# st.button('Prediction:')
# if prediction[0] == 1:
#     st.write('The model predicts that you have a heart disease.')
# else:
#     st.write('The model predicts that you do not have a heart disease.')

if st.button('Predict Heart Risk'):
    # Map categorical inputs to numerical values
    cp_value = cp_dict[cp]
    restecg_value = restecg_dict[restecg]
    slope_value = slope_dict[slope]
    thal_value = thal_dict[thal]

    # Prepare the input data for prediction
    data = np.array([age, sex, cp_value, trestbps, chol, fbs, restecg_value, thalach, exang, oldpeak, slope_value, ca, thal_value]).reshape(1, -1)

    # Make predictions using the model
    prediction = model.predict(data)

    # Display the prediction result
    st.subheader('Prediction:')
    if prediction[0] == 1:
        st.write('The model predicts that you have a heart disease.')
    else:
        st.write('The model predicts that you do not have a heart disease.')


