from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("model/car_price_model.pkl", "rb") as f:
    model, scaler = pickle.load(f)

# Smart suggestion system
def suggest_cars(price):
    if price < 500000:
        return [
            {"name": "Maruti Alto", "brand": "Maruti Suzuki",
             "img": "https://imgd.aeplcdn.com/600x337/n/cw/ec/40432/alto.jpeg"},
            {"name": "Renault Kwid", "brand": "Renault",
             "img": "https://imgd.aeplcdn.com/600x337/n/cw/ec/131179/kwid.jpeg"}
        ]

    elif price < 1000000:
        return [
            {"name": "Hyundai i20", "brand": "Hyundai",
             "img": "https://imgd.aeplcdn.com/600x337/n/cw/ec/40565/i20.jpeg"},
            {"name": "Tata Altroz", "brand": "Tata",
             "img": "https://imgd.aeplcdn.com/600x337/n/cw/ec/32597/altroz.jpeg"}
        ]

    elif price < 2000000:
        return [
            {"name": "Hyundai Creta", "brand": "Hyundai",
             "img": "https://imgd.aeplcdn.com/600x337/n/cw/ec/106815/creta.jpeg"},
            {"name": "Kia Seltos", "brand": "Kia",
             "img": "https://imgd.aeplcdn.com/600x337/n/cw/ec/144185/seltos.jpeg"}
        ]

    else:
        return [
            {"name": "BMW 3 Series", "brand": "BMW",
             "img": "https://imgd.aeplcdn.com/600x337/n/cw/ec/37640/3-series.jpeg"},
            {"name": "Audi A4", "brand": "Audi",
             "img": "https://imgd.aeplcdn.com/600x337/n/cw/ec/51909/a4.jpeg"}
        ]


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        hp = float(request.form['horsepower'])
        eng = float(request.form['enginesize'])
        city = float(request.form['citympg'])
        highway = float(request.form['highwaympg'])

        input_data = np.array([[hp, eng, city, highway]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]

        cars = suggest_cars(prediction)

        return render_template(
            "index.html",
            prediction_text=f"₹ {round(prediction, 2)}",
            car_suggestions=cars
        )

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
