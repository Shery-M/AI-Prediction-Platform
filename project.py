import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="AI Data Analyzer", layout="wide", page_icon="🤖")


st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 AI Data Analyzer & Predictor")
st.markdown("---")


st.sidebar.title("⚙️ Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file:
  
    df = pd.read_csv(uploaded_file)
    
   
    df = df.fillna(df.median(numeric_only=True))

   
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Total Records", f"{len(df):,}")
    if 'median_house_value' in df.columns:
        col_m2.metric("Avg House Price", f"${df['median_house_value'].mean():,.0f}")
        col_m3.metric("Avg Income", f"${df['median_income'].mean():,.2f}")

   
    tab_data, tab_viz, tab_model = st.tabs(["📊 Data Explorer", "📈 Visual Insights", "⚙️ ML Model"])

    with tab_data:
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"📍 Rows: {df.shape[0]} | Columns: {df.shape[1]}")
        with c2:
            st.success("✅ Missing values handled automatically")

        
        if 'latitude' in df.columns and 'longitude' in df.columns:
            st.subheader("📍 Geospatial Distribution")
            st.map(df)

    with tab_viz:
        st.subheader("🔍 Statistical Analysis")
        numeric_cols = df.select_dtypes(include=np.number).columns
        
        col_v1, col_v2 = st.columns([1, 2])
        with col_v1:
            selected_col = st.selectbox("Select column for distribution", numeric_cols)
        
        fig, ax = plt.subplots()
        sns.histplot(df[selected_col], kde=True, color='#2e7d32', ax=ax)
        st.pyplot(fig)

        st.subheader("🔗 Feature Correlation")
        if len(numeric_cols) > 1:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="YlGnBu", ax=ax2)
            plt.tight_layout()
            st.pyplot(fig2)

    with tab_model:
        st.subheader("🚀 Model Training")
        
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            target = st.selectbox("Select Target (Label)", df.columns, index=len(df.columns)-2)
        with col_t2:
            problem_type = st.radio("Problem Type", ["Regression", "Classification"], horizontal=True)

        if st.button("Start AI Training"):
            with st.spinner("Processing features and training model..."):
                
                X = df.drop(columns=[target])
                y = df[target]
                
                
                X = pd.get_dummies(X, drop_first=True)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                if problem_type == "Classification":
                    model = RandomForestClassifier(n_estimators=100)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    score = accuracy_score(y_test, preds)
                    st.success(f"🏆 Model Accuracy: {score:.2%}")
                else:
                    model = RandomForestRegressor(n_estimators=100)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    rmse = np.sqrt(mean_squared_error(y_test, preds))
                    st.success(f"📉 Model RMSE: {rmse:,.2f}")

                
                st.session_state['model'] = model
                st.session_state['features'] = X.columns

        
        if 'model' in st.session_state:
            st.divider()
            st.subheader("🔮 Live Prediction Test")
            st.write("Enter values to get a real-time prediction:")
            
            
            input_cols = st.columns(4)
            user_data = {}
            for i, feat in enumerate(st.session_state['features'][:4]):
                with input_cols[i % 4]:
                    user_data[feat] = st.number_input(f"{feat}", value=float(df[feat].mean()))
            
            if st.button("Predict Now"):
                
                full_input = pd.DataFrame(0, index=[0], columns=st.session_state['features'])
                for k, v in user_data.items():
                    full_input[k] = v
                
                prediction = st.session_state['model'].predict(full_input)
                st.balloons()
                st.metric("Result", f"{prediction[0]:,.2f}")

else:
    st.info("👋 Welcome! Please upload a CSV file from the sidebar to start.")
   
