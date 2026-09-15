import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="Delivery Delay Predictor",
    page_icon="❄️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------------
# WINTER FOREST THEME
# -------------------------------------------------------
st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background:
            linear-gradient(
                rgba(9, 30, 38, 0.88),
                rgba(15, 54, 58, 0.92)
            );
        color: #F2FAFA;
    }

    /* Main container */
    .block-container {
        max-width: 950px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Main title */
    h1 {
        color: #F3FBFC !important;
        text-align: center;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }

    /* Headings */
    h2, h3 {
        color: #DCEFEA !important;
    }

    /* Normal text */
    p, label {
        color: #E5F3F1 !important;
    }

    /* Input labels */
    div[data-testid="stWidgetLabel"] p {
        color: #EAF7F5 !important;
        font-weight: 600;
    }

    /* Number input */
    div[data-baseweb="input"] {
        background-color: rgba(240, 250, 248, 0.95) !important;
        border-radius: 10px;
    }

    div[data-baseweb="input"] input {
        color: #173D3A !important;
    }

    /* Input sections */
    div[data-testid="stNumberInput"],
    div[data-testid="stSlider"] {
        background: rgba(255, 255, 255, 0.06);
        padding: 12px 14px;
        border-radius: 14px;
        margin-bottom: 8px;
        border: 1px solid rgba(210, 240, 235, 0.12);
    }

    /* Button */
    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        border: none;
        background: linear-gradient(
            90deg,
            #315B55,
            #497C72
        );
        color: white;
        font-size: 18px;
        font-weight: 700;
        padding: 0.7rem 1rem;
        transition: 0.3s;
    }

    div.stButton > button:hover {
        background: linear-gradient(
            90deg,
            #47766E,
            #60958A
        );
        transform: translateY(-2px);
        color: white;
    }

    /* Metric/result boxes */
    div[data-testid="stAlert"] {
        border-radius: 14px;
    }

    /* Hide Streamlit footer */
    footer {
        visibility: hidden;
    }

    /* Forest decorative header */
    .forest-header {
        text-align: center;
        font-size: 30px;
        margin-bottom: -10px;
        letter-spacing: 4px;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #CFE4DF;
        font-size: 17px;
        margin-bottom: 28px;
    }

    /* Small card */
    .info-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        padding: 16px 18px;
        border-radius: 16px;
        margin-bottom: 20px;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------
# DECORATIVE HEADER
# -------------------------------------------------------
st.markdown(
    """
    <div class="forest-header">
        ❄️ 🌲 ❄️ 🌲 ❄️
    </div>
    """,
    unsafe_allow_html=True
)

st.title("Delivery Delay Predictor")

st.markdown(
    """
    <div class="subtitle">
        Predict shipment delays using delivery, traffic, weather,
        vehicle and operational conditions.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-card">
        🌲 <b>Winter Logistics Intelligence</b><br>
        Enter the shipment details below and let the model estimate
        the probability of a delivery delay.
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "logi.sav"

    if not model_path.exists():
        raise FileNotFoundError(
            "logi.sav was not found. Make sure the model file "
            "is uploaded to the same GitHub folder as app.py."
        )

    return joblib.load(model_path)


try:
    model = load_model()

except Exception as e:
    st.error("❄️ The prediction model could not be loaded.")
    st.code(str(e))
    st.info(
        "Make sure `logi.sav` is uploaded to the same GitHub "
        "repository folder as `app.py`."
    )
    st.stop()

# -------------------------------------------------------
# USER INPUT SECTION
# -------------------------------------------------------
st.header("❄️ Shipment Details")

col1, col2 = st.columns(2)

with col1:

    delivery_distance = st.number_input(
        "Delivery Distance (miles)",
        min_value=0.0,
        max_value=500.0,
        value=25.0,
        step=1.0
    )

    traffic_congestion = st.slider(
        "Traffic Congestion Level",
        min_value=1,
        max_value=5,
        value=3,
        help="1 = Very Low, 5 = Very High"
    )

    weather_condition = st.slider(
        "Weather Condition Score",
        min_value=1,
        max_value=5,
        value=2,
        help="Use the same coding used in your training dataset."
    )

    delivery_slot = st.slider(
        "Delivery Slot",
        min_value=1,
        max_value=4,
        value=2
    )

    driver_experience = st.number_input(
        "Driver Experience (Years)",
        min_value=0,
        max_value=50,
        value=5
    )

    num_stops = st.number_input(
        "Number of Stops",
        min_value=0,
        max_value=50,
        value=3
    )


with col2:

    vehicle_age = st.number_input(
        "Vehicle Age (Years)",
        min_value=0,
        max_value=30,
        value=4
    )

    road_condition_score = st.slider(
        "Road Condition Score",
        min_value=1,
        max_value=5,
        value=3,
        help="Use the same interpretation used during model training."
    )

    package_weight = st.number_input(
        "Package Weight (lbs)",
        min_value=0.0,
        max_value=500.0,
        value=15.0,
        step=0.5
    )

    fuel_efficiency = st.number_input(
        "Fuel Efficiency (mpg)",
        min_value=1.0,
        max_value=100.0,
        value=15.0,
        step=0.5
    )

    warehouse_processing_time = st.number_input(
        "Warehouse Processing Time (mins)",
        min_value=0,
        max_value=1440,
        value=45
    )

# -------------------------------------------------------
# CREATE INPUT DATA
# -------------------------------------------------------
input_data = pd.DataFrame(
    [{
        "Delivery_Distance": delivery_distance,
        "Traffic_Congestion": traffic_congestion,
        "Weather_Condition": weather_condition,
        "Delivery_Slot": delivery_slot,
        "Driver_Experience": driver_experience,
        "Num_Stops": num_stops,
        "Vehicle_Age": vehicle_age,
        "Road_Condition_Score": road_condition_score,
        "Package_Weight": package_weight,
        "Fuel_Efficiency": fuel_efficiency,
        "Warehouse_Processing_Time": warehouse_processing_time
    }]
)

# -------------------------------------------------------
# MATCH MODEL FEATURE ORDER
# -------------------------------------------------------
if hasattr(model, "feature_names_in_"):

    expected_features = list(model.feature_names_in_)

    missing_features = [
        feature
        for feature in expected_features
        if feature not in input_data.columns
    ]

    if missing_features:
        st.error(
            "The app inputs do not match the variables used "
            "while training the model."
        )

        st.write("Missing features:")
        st.write(missing_features)

        st.stop()

    input_data = input_data[expected_features]

# -------------------------------------------------------
# PREDICTION
# -------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("❄️ Predict Delivery Status"):

    try:

        prediction = model.predict(input_data)[0]

        # Probability
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]

            classes = list(model.classes_)

            if 1 in classes:
                delay_index = classes.index(1)
                delay_probability = probabilities[delay_index]
            else:
                delay_probability = probabilities[-1]

        else:
            delay_probability = None

        st.markdown("---")
        st.subheader("🌲 Prediction Result")

        # ---------------------------------------------------
        # DELAY
        # ---------------------------------------------------
        if prediction == 1:

            st.error("🚨 Predicted Status: DELIVERY DELAY")

            if delay_probability is not None:

                st.metric(
                    label="Estimated Delay Probability",
                    value=f"{delay_probability:.1%}"
                )

                st.progress(float(delay_probability))

            st.warning(
                "Consider reviewing traffic conditions, weather, "
                "warehouse processing time and the delivery route."
            )

        # ---------------------------------------------------
        # ON TIME
        # ---------------------------------------------------
        else:

            st.success("✅ Predicted Status: ON-TIME DELIVERY")

            if delay_probability is not None:

                on_time_probability = 1 - delay_probability

                st.metric(
                    label="Estimated On-Time Probability",
                    value=f"{on_time_probability:.1%}"
                )

                st.progress(float(on_time_probability))

            st.info(
                "Current shipment conditions indicate a relatively "
                "low risk of delivery delay."
            )

    except Exception as e:

        st.error("Prediction could not be generated.")

        st.write(
            "This usually happens when the variables in the app "
            "do not exactly match those used while training the model."
        )

        st.code(str(e))

# -------------------------------------------------------
# FOOTER
# -------------------------------------------------------
st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#BCD6D0;
        font-size:13px;
    ">
        ❄️ Delivery Intelligence • Winter Forest Edition 🌲
    </div>
    """,
    unsafe_allow_html=True
)
