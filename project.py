import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="AI App", layout="wide")

st.title("🤖 AI Data Analyzer & Predictor")

# Sidebar
st.sidebar.title("⚙️ Settings")

uploaded_file = st.file_uploader("Upload your dataset (CSV)")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # Missing values
    st.subheader("🧹 Missing Values")
    st.write(df.isnull().sum())

    # Visualization
    st.subheader("📈 Data Visualization")

    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) > 0:
        col = st.selectbox("Select column for histogram", numeric_cols)

        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numeric columns available for plotting")

    # Correlation
    st.subheader("🔗 Correlation Heatmap")
    if len(numeric_cols) > 1:
        fig2, ax2 = plt.subplots()
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)

    # Model
    st.subheader("⚙️ Train Model")

    target = st.selectbox("Select Target Column", df.columns)
    problem_type = st.radio("Problem Type", ["Classification", "Regression"])

    if st.button("Train Model"):

        with st.spinner("Training model..."):

            X = df.drop(columns=[target])
            y = df[target]

            X = pd.get_dummies(X, drop_first=True)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            if problem_type == "Classification":
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)

                st.success(f"✅ Accuracy: {acc:.2f}")

            else:
                model = RandomForestRegressor()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, preds))

                st.success(f"✅ RMSE: {rmse:.2f}")