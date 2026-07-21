import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# -----------------------------------------------------------------------------
# 1. Data Loading and Preprocessing Functions
# -----------------------------------------------------------------------------

@st.cache_data
def load_and_preprocess_data(file_path="Card.csv"):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # 1. Remove duplicates
    df = df.drop_duplicates()
    
    # Target column check
    if 'selling_price' not in df.columns:
        st.error("Target column 'selling_price' not found in the dataset.")
        st.stop()
        
    # 3. Remove unnecessary columns (Name has too much high cardinality for basic LR)
    if 'name' in df.columns:
        df = df.drop(columns=['name'])
        
    # 4. Convert columns with units to numeric values
    def extract_numeric(val):
        if pd.isna(val) or str(val).strip() == '':
            return np.nan
        # Regex to find the first sequence of digits (including decimals)
        match = re.search(r'([0-9]+\.?[0-9]*)', str(val))
        if match:
            return float(match.group(1))
        return np.nan

    unit_columns = ['mileage', 'engine', 'max_power', 'torque']
    for col in unit_columns:
        if col in df.columns:
            df[col] = df[col].apply(extract_numeric)
            
    # Separate features and target
    X = df.drop(columns=['selling_price'])
    y = df['selling_price']
    
    return X, y

@st.cache_resource
def train_model(X, y):
    # Identify numerical and categorical columns automatically
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 2. Handle missing values & 5. Encode categorical columns
    # We use pipelines to handle imputation and scaling/encoding simultaneously
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 6 & 7. Keep selling_price as target, Train Linear Regression
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Train the model on the full cleaned dataset
    model_pipeline.fit(X, y)
    
    return model_pipeline, numeric_features, categorical_features

# -----------------------------------------------------------------------------
# 2. Streamlit Application UI
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Car Price Predictor", layout="centered")
    st.title("🚗 Car Price Prediction App")
    st.write("This app uses for the Car Prediction price")
    
    # Load data and train model
    try:
        X, y = load_and_preprocess_data("Card.csv")
        model, num_cols, cat_cols = train_model(X, y)
    except FileNotFoundError:
        st.error("Error: `Card.csv` not found in the current directory. Please upload or ensure the file exists.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during data processing: {e}")
        st.stop()

    st.header("Enter Vehicle Specifications")
    
    # Generate input widgets dynamically based on final features
    input_data = {}
    
    # Create two columns for better UI layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Numerical Features")
        for feature in num_cols:
            # Determine appropriate step and min/max based on training data
            min_val = float(X[feature].min())
            max_val = float(X[feature].max())
            median_val = float(X[feature].median())
            
            # Format step for integers vs floats
            step = 1.0 if X[feature].dtype == 'float64' else 1
            
            input_data[feature] = st.number_input(
                label=feature.replace('_', ' ').title(),
                min_value=min_val,
                max_value=max_val,
                value=median_val,
                step=float(step)
            )
            
    with col2:
        st.subheader("Categorical Features")
        for feature in cat_cols:
            # Dropna to avoid passing NaN as a category choice
            unique_options = X[feature].dropna().unique().tolist()
            input_data[feature] = st.selectbox(
                label=feature.replace('_', ' ').title(),
                options=unique_options
            )

    # Prediction Action
    st.markdown("---")
    if st.button("Predict Selling Price", type="primary"):
        # Convert user input into a DataFrame
        user_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(user_df)[0]
        
        # Display the result
        st.success("### Estimated Selling Price")
        
        # Ensure prediction is non-negative
        if prediction < 0:
            st.warning("The model predicted a negative value, which is unrealistic. Consider revising the input parameters or checking the model constraints.")
            st.metric(label="Price (INR)", value="₹0")
        else:
            st.metric(label="Price (INR)", value=f"₹ {prediction:,.2f}")

if __name__ == "__main__":
    main()