import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("🔋 EV Battery Prediction Web App")
st.write("Predict battery consumption and range for your electric vehicle")

# Inputs
speed = st.slider("Speed (km/h)", 20, 120, 60)
distance = st.number_input("Distance to travel (km)", 1, 500, 50)
current_battery = st.slider("Current Battery (%)", 1, 100, 80)

# Prediction
if st.button("Predict Battery Usage"):
    input_features = np.array([[speed, distance]])
    predicted_usage = model.predict(input_features)[0]
    
    remaining_battery = current_battery - predicted_usage
    remaining_battery = max(0, round(remaining_battery, 2))

    st.subheader(" Prediction Results")
    st.write(f" **Predicted Battery Usage:** {round(predicted_usage,2)} %")
    st.write(f" **Remaining Battery After Trip:** {remaining_battery} %")

    if remaining_battery > 20:
        st.success("Battery is sufficient for your trip!")
    else:
        st.warning("Low battery! You may need to charge before travelling.")

