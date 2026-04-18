# ================================
# FLASK WEB APP (FINAL)
# ================================

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("model/car_price_model.pkl", "rb") as f:
    model, scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        horsepower = float(request.form['horsepower'])
        enginesize = float(request.form['enginesize'])
        citympg = float(request.form['citympg'])
        highwaympg = float(request.form['highwaympg'])

        # Prepare input
        input_data = np.array([[horsepower, enginesize, citympg, highwaympg]])

        # Scale
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]

        return render_template(
            "index.html",
            prediction_text=f"Estimated Car Price: ₹ {round(prediction, 2)}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )
def suggest_cars(price):
    if price < 500000:
        return ["Maruti Alto", "Renault Kwid", "Datsun Redi-GO"]

    elif price < 1000000:
        return ["Maruti Swift", "Hyundai i20", "Tata Altroz"]

    elif price < 2000000:
        return ["Hyundai Creta", "Kia Seltos", "Honda City"]

    elif price < 4000000:
        return ["Toyota Innova", "Mahindra XUV700", "MG Hector"]

    else:
        return ["BMW 3 Series", "Audi A4", "Mercedes-Benz C-Class"]


if __name__ == "__main__":
    app.run(debug=True)