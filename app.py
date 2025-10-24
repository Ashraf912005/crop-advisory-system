import streamlit as st
import pandas as pd
import os
import joblib
import gzip  # <-- for manual compression
import shutil
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -----------------------------------------------
# 🔹 Helper: Compress .pkl files manually using gzip
# -----------------------------------------------
def compress_pickle(input_path, target_size_mb=25):
    """
    Compress a .pkl file using gzip. Keeps only one final .pkl file.
    Example: crop_recommendation.pkl -> compressed crop_recommendation.pkl
    """
    temp_path = input_path + ".gz"

    # Compress with gzip
    with open(input_path, "rb") as f_in:
        with gzip.open(temp_path, "wb", compresslevel=9) as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Get compressed size
    compressed_size = os.path.getsize(temp_path) / (1024 * 1024)

    # If smaller than target size, replace original
    if float(compressed_size) <= float(target_size_mb):
        os.remove(input_path)
        os.rename(temp_path, input_path)
        print(f"✅ Compressed {input_path} → {compressed_size:.2f} MB")
    else:
        os.remove(temp_path)
        print(f"⚠️ Compression complete but file still {compressed_size:.2f} MB (limit {target_size_mb} MB)")




# -----------------------------------------------
# 🔹 Function to train and save model if not found
# -----------------------------------------------
def train_and_save_model():
    st.warning("⚠️ Model file not found — training new model. Please wait...")

    # Load dataset
    df = pd.read_csv("Maharashtra_crop_dataset.csv")
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    X = df[[
        "season",
        "district",
        "soiltype",
        "avgrainfall_mm",
        "avgtemp_c",
        "avghumidity_%",
        "soil_ph",
        "nitrogen_kg_ha",
        "phosphorus_kg_ha",
        "potassium_kg_ha"
    ]]
    y = df["Crop"]

    # Encode categorical features
    X = pd.get_dummies(X, columns=["district", "soiltype", "season"], drop_first=True)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train RandomForest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model (compressed)
    joblib.dump(model, "crop_recommendation.pkl", compress=9)
    joblib.dump(X.columns.tolist(), "model_columns.pkl", compress=9)

    st.success("✅ Model trained and saved successfully!")

    return model, X.columns.tolist(), df


# -----------------------------------------------
# 🔹 Load model and columns (train if missing)
# -----------------------------------------------
@st.cache_resource
def load_model_and_columns():
    if not os.path.exists("crop_recommendation.pkl") or not os.path.exists("model_columns.pkl"):
        model, model_columns, df = train_and_save_model()
    else:
        model = joblib.load("crop_recommendation.pkl")
        model_columns = joblib.load("model_columns.pkl")
        df = pd.read_csv("Maharashtra_crop_dataset.csv")
        df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    return model, model_columns, df


model, model_columns, df = load_model_and_columns()

# -----------------------------------------------
# 🔹 App UI
# -----------------------------------------------
st.title("🌾 Maharashtra Crop Recommendation System")
st.write("Enter your soil and weather conditions below to get a recommended crop and weather alerts!")

available_districts = sorted(df["district"].unique())
available_soiltypes = sorted(df["soiltype"].unique())
available_seasons = sorted(df["season"].unique())

# -----------------------------------------------
# 🔹 Input Form
# -----------------------------------------------
with st.form("crop_form"):
    st.subheader("🧾 Enter Input Data")

    col1, col2, col3 = st.columns(3)
    with col1:
        district = st.selectbox("District", available_districts)
        soiltype = st.selectbox("Soil Type", available_soiltypes)
        season = st.selectbox("Season", available_seasons)

    with col2:
        avgrainfall_mm = st.number_input("Average Rainfall (mm)", min_value=0.0, step=1.0)
        avgtemp_c = st.number_input("Average Temperature (°C)", min_value=0.0, step=0.1)
        avghumidity = st.number_input("Average Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)

    with col3:
        soil_ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, step=0.1)
        nitrogen = st.number_input("Nitrogen (kg/ha)", min_value=0.0, step=1.0)
        phosphorus = st.number_input("Phosphorus (kg/ha)", min_value=0.0, step=1.0)
        potassium = st.number_input("Potassium (kg/ha)", min_value=0.0, step=1.0)

    submitted = st.form_submit_button("🔍 Predict Crop")

# -----------------------------------------------
# 🔹 Weather Alert Logic
# -----------------------------------------------
def get_weather_alert(temp, humidity, rainfall):
    alerts = []
    if temp > 35:
        alerts.append("🌡️ High temperature may cause heat stress to crops.")
    elif temp < 15:
        alerts.append("❄️ Low temperature — growth may slow down.")

    if humidity > 85:
        alerts.append("💧 Excess humidity — possible risk of fungal diseases.")
    elif humidity < 30:
        alerts.append("🔥 Low humidity — irrigation may be required frequently.")

    if rainfall > 1200:
        alerts.append("☔ Heavy rainfall — ensure proper drainage.")
    elif rainfall < 400:
        alerts.append("🌤️ Low rainfall — consider drought-tolerant crops or irrigation support.")

    if not alerts:
        alerts.append("✅ Weather conditions look favorable for most crops.")
    return alerts


# -----------------------------------------------
# 🔹 Expected Yield Logic (approx, per hectare)
# -----------------------------------------------
def get_expected_yield(crop_name):
    yield_data = {
        "Cotton": "18–25 quintals/ha",
        "Soybean": "20–30 quintals/ha",
        "Tur": "10–15 quintals/ha",
        "Wheat": "35–45 quintals/ha",
        "Jowar": "20–30 quintals/ha",
        "Rice": "40–55 quintals/ha",
        "Gram": "12–18 quintals/ha",
        "Sugarcane": "800–1000 quintals/ha",
        "Maize": "40–50 quintals/ha",
        "Groundnut": "20–25 quintals/ha",
    }
    return yield_data.get(crop_name, "Data not available")


# -----------------------------------------------
# 🔹 Prediction Logic
# -----------------------------------------------
if submitted:
    try:
        user_data = pd.DataFrame([{
            "district": district,
            "soiltype": soiltype,
            "season": season,
            "avgrainfall_mm": avgrainfall_mm,
            "avgtemp_c": avgtemp_c,
            "avghumidity_%": avghumidity,
            "soil_ph": soil_ph,
            "nitrogen_kg_ha": nitrogen,
            "phosphorus_kg_ha": phosphorus,
            "potassium_kg_ha": potassium
        }])

        # One-hot encode and align columns
        user_data = pd.get_dummies(user_data, columns=["district", "soiltype", "season"], drop_first=True)
        user_data = user_data.reindex(columns=model_columns, fill_value=0)

        # Predict crop
        prediction = model.predict(user_data)[0]
        yield_est = get_expected_yield(prediction)
        alerts = get_weather_alert(avgtemp_c, avghumidity, avgrainfall_mm)

        # Display results
        st.success(f"✅ **Recommended Crop:** {prediction}")
        st.info(f"🌾 **Expected Yield:** {yield_est}")

        st.subheader("🌤️ Weather Alerts")
        for alert in alerts:
            st.write(alert)

        with st.expander("📋 Input Summary"):
            st.json({
                "District": district,
                "Soil Type": soiltype,
                "Season": season,
                "Average Rainfall (mm)": avgrainfall_mm,
                "Temperature (°C)": avgtemp_c,
                "Humidity (%)": avghumidity,
                "Soil pH": soil_ph,
                "Nitrogen (kg/ha)": nitrogen,
                "Phosphorus (kg/ha)": phosphorus,
                "Potassium (kg/ha)": potassium
            })

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
