import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Title of the app
st.title("Medical Condition Classification App")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)
    
    # Show the first few rows of the data
    st.write("Data Preview:")
    st.write(data.head())

    # Drop unnecessary columns
    data = data.drop(columns=['id', 'full_name'])

    # Handle missing values
    imputer_median = SimpleImputer(strategy='median')
    imputer_mean = SimpleImputer(strategy='mean')

    data['age'] = imputer_median.fit_transform(data[['age']])
    data['bmi'] = imputer_median.fit_transform(data[['bmi']])
    data['blood_pressure'] = imputer_mean.fit_transform(data[['blood_pressure']])
    data['glucose_levels'] = imputer_mean.fit_transform(data[['glucose_levels']])

    # Encode categorical features
    label_encoder = LabelEncoder()
    data['gender'] = label_encoder.fit_transform(data['gender'])
    data['smoking_status'] = label_encoder.fit_transform(data['smoking_status'])
    data['condition'] = label_encoder.fit_transform(data['condition'])

    # Split the dataset into features (X) and target (y)
    X = data.drop(columns=['condition'])
    y = data['condition']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Display the evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {accuracy:.2f}")

    st.write("Classification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.write(report_df)

    # Sidebar options to explore different feature combinations
    st.sidebar.title("Feature Exploration")
    feature = st.sidebar.selectbox("Select a feature to explore:", X.columns)

    # Display a breakdown of the selected feature
    st.write(f"Feature Analysis: {feature}")
    st.write(data[feature].value_counts())

# Instructions when no file is uploaded
else:
    st.write("Please upload a CSV file to proceed.")
