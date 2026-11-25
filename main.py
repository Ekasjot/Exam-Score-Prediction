
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Exam Score Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Exam Score Prediction Dashboard")

st.write(
    "Predict a student's final exam score based on study habits, "
    "attendance, and sleep metrics using an optimized XGBoost model."
)


# Load trained model and encoders
@st.cache_resource
def load_artifacts():

    with open("label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    import xgboost as xgb

    model = xgb.XGBRegressor()
    model.load_model("xgboost_model.json")

    return encoders, model


try:
    encoders, model = load_artifacts()

except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.info(
        "Make sure label_encoders.pkl and "
        "xgboost_model.pkl are in your project folder."
    )
    st.stop()


# User inputs
col1, col2 = st.columns(2)

with col1:

    st.subheader("📚 Study & Performance")

    study_hours = st.slider(
        "Study Hours (Daily)",
        0.0, 24.0, 4.0, 0.1
    )

    class_attendance = st.slider(
        "Class Attendance (%)",
        0.0, 100.0, 85.0, 0.5
    )

    selected_study_method = st.selectbox(
        "Preferred Study Method",
        encoders["study_method"].classes_
    )


with col2:

    st.subheader("🛌 Health & Environment")

    sleep_hours = st.slider(
        "Sleep Hours (Average)",
        0.0, 24.0, 7.0, 0.1
    )

    selected_sleep_quality = st.selectbox(
        "Sleep Quality",
        encoders["sleep_quality"].classes_
    )

    selected_facility_rating = st.selectbox(
        "Facility Rating",
        encoders["facility_rating"].classes_
    )


st.divider()


# Prediction
if st.button("🔮 Predict Exam Score", use_container_width=True):

    try:

        # Encode categorical inputs
        encoded_sleep_quality = encoders[
            "sleep_quality"
        ].transform([selected_sleep_quality])[0]

        encoded_study_method = encoders[
            "study_method"
        ].transform([selected_study_method])[0]

        encoded_facility_rating = encoders[
            "facility_rating"
        ].transform([selected_facility_rating])[0]


        # Create input dataframe
        input_data = pd.DataFrame({

            "study_hours": [study_hours],

            "class_attendance": [class_attendance],

            "sleep_hours": [sleep_hours],

            "sleep_quality": [encoded_sleep_quality],

            "study_method": [encoded_study_method],

            "facility_rating": [encoded_facility_rating]

        })


        # Verify model feature compatibility
        expected_features = getattr(
            model, "feature_names_in_", None
        )

        if expected_features is not None:

            if set(expected_features) != set(input_data.columns):

                st.error(
                    "Input features do not match the features "
                    "used during model training."
                )

                st.write("Model expects:", list(expected_features))
                st.write("App provides:", list(input_data.columns))

                st.stop()

            input_data = input_data[list(expected_features)]


        # Predict
        prediction = model.predict(input_data)[0]

        final_score = np.clip(prediction, 0, 100)

        st.success(
            f"🎉 Predicted Exam Score: {final_score:.2f} / 100"
        )


        # Performance feedback
        if final_score >= 85:

            st.balloons()
            st.info("Excellent performance expected! 🌟")

        elif final_score >= 50:

            st.info("On track to pass! Keep it up. 👍")

        else:

            st.warning(
                "Consider improving your study habits "
                "and attendance."
            )


    except Exception as e:

        st.error(f"Prediction error: {e}")
