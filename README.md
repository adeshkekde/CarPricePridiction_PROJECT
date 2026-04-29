# 🚗 Car Price Prediction & Recommendation System

A machine learning web application that predicts car prices based on key features and recommends similar cars using a dataset-driven approach.

---

## 📌 Project Overview

This project uses a **Machine Learning Regression Model** to estimate the price of a car based on input features like:

* Horsepower
* Engine Size
* City MPG
* Highway MPG

After predicting the price, the system **recommends similar cars from the dataset** based on price proximity.

---

## 🚀 Features

* 🔮 Predict car price using ML (Random Forest)
* 📊 Data preprocessing & feature scaling
* 🌐 Web interface using Flask
* 🤖 Dataset-based car recommendations
* 🎨 Modern responsive UI
* 📁 Clean project structure

---

## 🧠 Machine Learning Workflow

1. Data Collection (Car dataset)
2. Data Cleaning & Preprocessing
3. Feature Selection
4. Model Training (Random Forest Regressor)
5. Model Evaluation
6. Deployment using Flask

---

## 📂 Project Structure

```
CarPriceWebApp/
│
├── data/
│   └── car data.csv
│
├── model/
│   └── car_price_model.pkl
│
├── templates/
│   └── index.html
│
├── static/ (optional)
│
├── app.py
├── train.py
└── requirements.txt
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/CarPriceWebApp.git
cd CarPriceWebApp
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train the model

```bash
python train.py
```

### 4️⃣ Run the application

```bash
python app.py
```

### 5️⃣ Open in browser

```
http://127.0.0.1:5000/
```

---

## 📊 Technologies Used

* Python 🐍
* Pandas
* NumPy
* Scikit-learn
* Flask
* HTML, CSS

---

## 📈 Model Details

* Algorithm: **Random Forest Regressor**
* Input Features:

  * Horsepower
  * Engine Size
  * City MPG
  * Highway MPG
* Output:

  * Predicted Car Price

---

## 🚗 Recommendation System

* Uses dataset filtering based on predicted price
* Selects cars within a price range (±20%)
* Returns top matching cars

---

## 🎯 Use Cases

* Car price estimation tools
* Automobile recommendation systems
* ML-based decision support systems

---

## 📌 Future Improvements

* 🔍 Advanced ML-based recommendation (KNN / similarity)
* 📊 Data visualization dashboard
* 🌐 Deployment on cloud (Render / Heroku)
* 🔐 User login system
* 📱 Mobile-friendly UI enhancements

---

## 👨‍💻 Author
GitHub: https://github.com/adeshkekde


This project is open-source and available under the MIT License.
