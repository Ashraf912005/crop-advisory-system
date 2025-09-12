# ---------------------------
# crop_predictor_app_debug.py
# Robust Streamlit crop predictor with uploader, diagnostics, and detailed error reporting
# ---------------------------

import os
import io
import traceback
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Crop Predictor (Debug)", layout="wide")

# -----------------------
# Utilities
# -----------------------
REQUIRED_COLS = ['Season', 'State', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield', 'Crop']

def make_sample_df():
    # small synthetic sample for download / quick test
    data = {
        'Season': ['Kharif','Kharif','Rabi','Rabi','Kharif','Rabi'],
        'State' : ['state1','state1','state1','state2','state2','state2'],
        'Area': [1.0, 2.5, 1.2, 0.8, 3.0, 1.5],
        'Annual_Rainfall': [800, 900, 400, 500, 1000, 450],
        'Fertilizer': [100, 120, 90, 80, 130, 95],
        'Pesticide': [2.0, 1.5, 1.8, 1.0, 2.2, 1.1],
        'Yield': [2.5, 3.0, 1.5, 1.3, 3.5, 1.6],
        'Crop': ['CropA','CropA','CropB','CropC','CropA','CropB']
    }
    return pd.DataFrame(data)

# -----------------------
# 1) DATA INPUT / UPLOAD
# -----------------------
st.title("🌱 Crop Predictor — Debug Friendly")

st.markdown("Upload your `crop_yield.csv` or use the sample provided for quick testing.")

uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
use_local = st.checkbox("Use local file 'crop_yield.csv' (if present)", value=False)

df = None
load_error = None

if uploaded is not None:
    try:
        uploaded.seek(0)
        df = pd.read_csv(uploaded)
    except Exception as e:
        load_error = f"Error reading uploaded CSV: {e}\n\nTrace:\n{traceback.format_exc()}"
elif use_local and os.path.exists("crop_yield.csv"):
    try:
        df = pd.read_csv("crop_yield.csv")
    except Exception as e:
        load_error = f"Error reading local CSV: {e}\n\nTrace:\n{traceback.format_exc()}"

if df is None and not load_error:
    st.info("No file provided. You can download & use a sample dataset for testing.")
    sample = make_sample_df()
    csv_bytes = sample.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download sample CSV", data=csv_bytes, file_name="crop_yield_sample.csv", mime="text/csv")
    st.stop()

if load_error:
    st.error(load_error)
    st.stop()

# -----------------------
# 2) PREPROCESS & VALIDATE
# -----------------------
st.subheader("Dataset diagnostics")
st.write("Shape:", df.shape)
st.write("Columns:", list(df.columns))
st.write("Missing values per column:")
st.write(df.isnull().sum())

# show head and dtypes
with st.expander("Show head & dtypes"):
    st.dataframe(df.head())
    st.write(df.dtypes)

# check required columns
missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    st.error(f"CSV missing required columns: {missing}")
    st.stop()

# keep required columns only
df = df[REQUIRED_COLS].copy()

# normalize text columns
df['Season'] = df['Season'].astype(str).str.strip().str.lower()
df['State']  = df['State'].astype(str).str.strip().str.lower()
df['Crop']   = df['Crop'].astype(str).str.strip()

# ensure numeric columns are numeric
for col in ['Area','Annual_Rainfall','Fertilizer','Pesticide','Yield']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
# drop rows with numeric NaNs
before_drop = df.shape[0]
df = df.dropna(subset=['Area','Annual_Rainfall','Fertilizer','Pesticide','Yield'])
after_drop = df.shape[0]
st.write(f"Dropped {before_drop - after_drop} rows due to non-numeric values in numeric columns.")

if df.empty:
    st.error("No usable rows after preprocessing (numeric conversion / NaN drop). Check your CSV.")
    st.stop()

# quick distribution visuals
with st.expander("Basic distributions (Area, Rainfall, Yield)"):
    fig, ax = plt.subplots(1,3, figsize=(15,4))
    df['Area'].hist(ax=ax[0]); ax[0].set_title("Area")
    df['Annual_Rainfall'].hist(ax=ax[1]); ax[1].set_title("Annual_Rainfall")
    df['Yield'].hist(ax=ax[2]); ax[2].set_title("Yield")
    st.pyplot(fig)

# -----------------------
# 3) STATE SELECTION
# -----------------------
states = sorted(df['State'].unique())
if not states:
    st.error("No states available in dataset.")
    st.stop()

state_display = [s.title() for s in states]
selected_state_display = st.selectbox("Select State", state_display)
selected_state = selected_state_display.lower()
df_state = df[df['State'] == selected_state].copy()
st.write("Rows for selected state:", df_state.shape[0])

if df_state.shape[0] < 2:
    st.warning("Very few rows for this state — predictions may be unreliable. Consider using larger dataset or a different state.")

# remove rare crops (>=2 samples), fallback to original if too aggressive
crop_counts = df_state['Crop'].value_counts()
valid_crops = crop_counts[crop_counts >= 2].index
if len(valid_crops) >= 1:
    df_state_filtered = df_state[df_state['Crop'].isin(valid_crops)].copy()
    if df_state_filtered.shape[0] >= 2:
        df_state = df_state_filtered
    else:
        st.info("Filtering rare crops left too few samples; using full state's data as fallback.")

# -----------------------
# 4) TRAIN MODEL (wrapped)
# -----------------------
st.subheader("Model training & diagnostics")

def train_and_evaluate(df_state):
    try:
        if df_state.empty:
            return {"error": "df_state is empty"}

        # OHE compat
        try:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

        X_cat = ohe.fit_transform(df_state[['Season']])
        X_num = df_state[['Area','Annual_Rainfall','Fertilizer','Pesticide','Yield']].astype(float).values
        X = np.hstack([X_cat, X_num])
        y = df_state['Crop'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # split (try stratify)
        try:
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=None)

        n_estimators = 1500
        if X_train.shape[0] < 50:
            n_estimators = 200

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)

        result = {}
        result['ohe'] = ohe
        result['scaler'] = scaler
        result['rf'] = rf
        result['X_test'] = X_test
        result['y_test'] = y_test
        result['X_train_shape'] = X_train.shape
        result['X_shape'] = X.shape
        # metrics
        if len(y_test) > 0:
            y_pred = rf.predict(X_test)
            result['accuracy'] = float(accuracy_score(y_test, y_pred))
            result['y_pred'] = y_pred
            result['labels'] = rf.classes_.tolist()
        else:
            result['accuracy'] = None
        return result

    except Exception as e:
        return {"error": traceback.format_exc()}

model_info = train_and_evaluate(df_state)

if model_info.get("error"):
    st.error("Model training failed. See details below.")
    st.exception(model_info["error"])
    st.stop()
else:
    acc = model_info['accuracy']
    if acc is not None:
        st.success(f"Trained. Accuracy on held-out test set: {acc:.4f}")
    else:
        st.info("Trained, but no test samples to evaluate accuracy (very small dataset).")

    # confusion matrix (if available)
    if model_info.get('y_test') is not None and len(model_info['y_test']) > 0:
        try:
            labels = model_info['labels']
            cm = confusion_matrix(model_info['y_test'], model_info['y_pred'], labels=labels)
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(cm, annot=True, fmt='d', ax=ax, xticklabels=labels, yticklabels=labels, cmap="Blues")
            ax.set_title(f"Confusion Matrix ({selected_state_display})")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
        except Exception as e:
            st.warning("Could not draw confusion matrix; see error below.")
            st.exception(traceback.format_exc())

# -----------------------
# 5) PREDICTION FORM
# -----------------------
st.subheader("Make a prediction")

available_seasons = sorted(df_state['Season'].unique())
if not available_seasons:
    st.error("No seasons available for selected state.")
    st.stop()

season_display = [s.title() for s in available_seasons]

with st.form("predict"):
    season_in = st.selectbox("Season", season_display)
    area_in = st.number_input("Area (hectares)", min_value=0.01, format="%.2f", value=1.00)
    rain_in = st.number_input("Annual Rainfall (mm)", min_value=0.0, format="%.1f", value=500.0)
    fert_in = st.number_input("Fertilizer (kg/ha)", min_value=0.0, format="%.1f", value=100.0)
    pest_in = st.number_input("Pesticide (kg/ha)", min_value=0.0, format="%.2f", value=1.0)
    yield_in = st.number_input("Expected Yield", min_value=0.0, format="%.2f", value=2.0)
    sub = st.form_submit_button("Predict")

if sub:
    try:
        ohe = model_info['ohe']
        scaler = model_info['scaler']
        rf = model_info['rf']
        season_to_use = season_in.strip().lower()
        # transform category
        sample_cat = ohe.transform([[season_to_use]])
        sample_num = np.array([[area_in, rain_in, fert_in, pest_in, yield_in]]).astype(float)
        sample = np.hstack([sample_cat, sample_num])
        sample_scaled = scaler.transform(sample)
        pred = rf.predict(sample_scaled)[0]
        st.success(f"🌾 Suggested Crop: **{pred}**")
        st.info(f"Area in acres: {(area_in * 2.47105):.2f} acres")
    except Exception as e:
        st.error("Prediction failed. See error details below.")
        st.exception(traceback.format_exc())

st.markdown("---")
st.caption("Tip: If you still see errors, open the 'Exception' trace shown above and paste it here. The trace gives exact line numbers and helps me fix the issue quickly.")


