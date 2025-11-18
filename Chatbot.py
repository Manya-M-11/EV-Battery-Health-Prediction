# EV Battery Life Prediction Chatbot
# Developed by Manya

def predict_battery_health(age, cycles, temp, drive_type):
    # Basic formula for prediction (can be adjusted)
    health = 100 - (age * 4) - (cycles * 0.03)

    # Temperature effect
    if temp > 35:
        health -= (temp - 35) * 0.7
    elif temp < 10:
        health -= (10 - temp) * 0.3

    # Driving condition effect
    if drive_type.lower() == "city":
        health -= 3  # more stop-start driving
    elif drive_type.lower() == "highway":
        health -= 1  # smoother drive, less stress

    # Ensure health stays within 0-100%
    if health < 0:
        health = 0
    elif health > 100:
        health = 100

    return health


def chatbot():
    print("🔋 Welcome to the EV Battery Life Prediction Chatbot! 🔋")
    print("Let's find out how healthy your EV battery is.\n")

    # Get user inputs
    battery_capacity = float(input("Enter your battery capacity (in kWh): "))
    age = float(input("Enter the age of your EV (in years): "))
    cycles = float(input("Enter total charging cycles completed: "))
    temp = float(input("Enter average operating temperature (°C): "))
    drive_type = input("Do you usually drive in 'city' or 'highway' conditions? ")

    # Predict battery health
    health = predict_battery_health(age, cycles, temp, drive_type)

    # Estimate remaining range
    estimated_range = (battery_capacity * health / 100) * 5  # simple assumption

    print("\n🔹 Prediction Results 🔹")
    print(f"Estimated Battery Health: {health:.2f}%")
    print(f"Estimated Driving Range: {estimated_range:.2f} km (approx.)")

    if health > 80:
        print("✅ Your battery is in great condition!")
    elif 50 <= health <= 80:
        print("⚠️ Your battery is moderately healthy. Consider regular maintenance.")
    else:
        print("🔋 Your battery health is low. Plan for replacement soon.")


# Run chatbot
if __name__ == "__main__":
    chatbot()
