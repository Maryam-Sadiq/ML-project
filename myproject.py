# streamlit_housing_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import io

# ----------------- App Title -----------------
st.subheader("==========================================")
st.subheader("-------------------------------------------------------------------")
st.subheader("*************Prepared by: Maryam ***************")
st.subheader("****************** AI-493009 ********************")
st.subheader("**** project submitted to: Miss Tayyaba Akram ****")
st.subheader("-------------------------------------------------------------------")
st.subheader("==========================================")

st.title("🏠 Housing Price Prediction App")
st.write("Analyzing and predicting house prices from a fixed CSV file.")

# ----------------- Load Dataset -----------------
# Upload your file
data = pd.read_csv("housing.csv")

st.subheader("Dataset Preview")
st.dataframe(data.head())

import streamlit as st

# ----------dictionary of column full form---------
full_forms = {
    "CRIM": "Per Capita Crime Rate by Town",
    "ZN": "Proportion of Residential Land Zoned for Large Plots",
    "INDUS": "Proportion of Non-Retail Business Acres per Town",
    "CHAS": "Charles River Dummy Variable",
    "NOX": "Nitric Oxide Concentration",
    "RM": "Average Number of Rooms per Dwelling",
    "AGE": "Proportion of Owner-Occupied Units Built Before 1940",
    "DIS": "Weighted Distance to Employment Centers",
    "RAD": "Index of Accessibility to Radial Highways",
    "TAX": "Full-Value Property Tax Rate",
    "PTRATIO": "Pupil–Teacher Ratio by Town",
    "B": "Proportion of Black Population",
    "LSTAT": "Percentage of Lower Status Population",
    "PRICE": "Median Value of Owner-Occupied Homes"
}

st.title("House Price Dataset – Column Full Forms")

for col in df.columns:
    st.write(f"*{col}* : {full_forms.get(col, 'Full form not found')}")

# ----------------- Dataset Info -----------------
st.subheader("Dataset Info")
buffer = io.StringIO()
data.info(buf=buffer)
info_df = buffer.getvalue()
st.text(info_df)

st.subheader("Dataset Description")
st.write(data.describe())

st.subheader("Dataset Shape & Missing Values")
st.write(f"Shape: {data.shape}")
st.write(data.isnull().sum())

st.subheader("Unique Values per Column")
st.write(data.nunique())

# Duplicate rows
st.subheader("Duplicate Rows")

duplicates = df[df.duplicated()]
st.write(duplicates)
st.write("duplicate rows",df.duplicate().sum())

# ----------------- Correlation Heatmap -----------------
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="Greens", ax=ax)
st.pyplot(fig)

# ----------------- Feature-Target Split -----------------
if 'PRICE' not in data.columns:
    st.error("Target column 'PRICE' not found in dataset!")
else:
    X = data.drop('PRICE', axis=1)
    y = data['PRICE']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # ----------------- Function to Display Metrics -----------------
    def display_metrics(y_true, y_pred):
        r2 = metrics.r2_score(y_true, y_pred)
        adj_r2 = 1 - (1-r2)*(len(y_true)-1)/(len(y_true)-X.shape[1]-1)
        mae = metrics.mean_absolute_error(y_true, y_pred)
        mse = metrics.mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        return {
            "R²": r2,
            "Adjusted R²": adj_r2,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse
        }

    # ----------------- Linear Regression -----------------
    st.subheader("Linear Regression Model")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_train_pred = lr.predict(X_train)
    y_test_pred = lr.predict(X_test)


    st.write("**Training Metrics:**", display_metrics(y_train, y_train_pred))
    st.write("**Test Metrics:**", display_metrics(y_test, y_test_pred))


   # ----------------- Linear Regression Coefficient -----------------
    st.subheader("Linear Regression Coefficients")
    coeffcients = pd.DataFrame([X_train.columns,lr.coef_]).T
    coeffcients = coeffcients.rename(columns={0: 'Attribute', 1: 'Coefficients'})
    st.dataframe(coeffcients)


    # Scatter plot: Actual vs Predicted
    st.subheader("Actual vs Predicted Prices (Linear Regression)")
    fig2, ax2 = plt.subplots()
    ax2.scatter(y_test, y_test_pred, alpha=0.7)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax2.set_xlabel("Actual Prices")
    ax2.set_ylabel("Predicted Prices")
    st.pyplot(fig2)

    # Residual plot
    st.subheader("Residual Plot (Linear Regression)")
    fig3, ax3 = plt.subplots()
    ax3.scatter(y_test_pred, y_test - y_test_pred, alpha=0.7)
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.set_xlabel("Predicted Prices")
    ax3.set_ylabel("Residuals")
    st.pyplot(fig3)

    # ----------------- Random Forest Regression -----------------
    st.subheader("Random Forest Regression Model")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_train_pred_rf = rf.predict(X_train)
    y_test_pred_rf = rf.predict(X_test)

    st.write("**Training Metrics:**", display_metrics(y_train, y_train_pred_rf))
    st.write("**Test Metrics:**", display_metrics(y_test, y_test_pred_rf))

    # ----------------- Random Forest: Actual vs Predicted -----------------
    st.subheader("Actual vs Predicted Prices (Random Forest)")
    fig_rf1, ax_rf1 = plt.subplots()
    ax_rf1.scatter(y_test, y_test_pred_rf, alpha=0.7)
    ax_rf1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax_rf1.set_xlabel("Actual Prices")
    ax_rf1.set_ylabel("Predicted Prices")
    st.pyplot(fig_rf1)

    # ----------------- Random Forest: Residual Plot -----------------
    st.subheader("Residual Plot (Random Forest)")
    fig_rf2, ax_rf2 = plt.subplots()
    ax_rf2.scatter(y_test_pred_rf, y_test - y_test_pred_rf, alpha=0.7)
    ax_rf2.axhline(y=0, color='r', linestyle='--')
    ax_rf2.set_xlabel("Predicted Prices")
    ax_rf2.set_ylabel("Residuals")
    st.pyplot(fig_rf2)

    # ----------------- Model Comparison -----------------
    models_df = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest"],
        "R² Score (%)": [
            metrics.r2_score(y_test, y_test_pred)*100,
            metrics.r2_score(y_test, y_test_pred_rf)*100
        ]
    }).sort_values(by="R² Score (%)", ascending=False)

    st.subheader("Model Comparison (R²)")
    st.dataframe(models_df)


