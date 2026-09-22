# 🏠 Household Energy Consumption Prediction

A Machine Learning project that predicts **household energy consumption (kWh)** using household and environmental factors such as household size, average temperature, and peak-hour energy usage.

The project uses **Polynomial Regression** to capture nonlinear relationships between the input features and energy consumption.

## 📌 Project Overview

Energy consumption can vary depending on several factors, including the number of people in a household, temperature, and peak-hour usage.

In this project, a **Polynomial Regression model (Degree 2)** is trained to predict:

> **Energy Consumption (kWh)**

### Input Features

* `Household_Size`
* `Avg_Temperature_C`
* `Peak_Hours_Usage_kWh`

### Target

* `Energy_Consumption_kWh`

---

## 📂 Dataset

The dataset contains **28,107 records** and **7 columns**.

| Column                   | Description                                            |
| ------------------------ | ------------------------------------------------------ |
| `Household_ID`           | Unique household identifier                            |
| `Date`                   | Date of energy consumption                             |
| `Energy_Consumption_kWh` | Total energy consumption                               |
| `Household_Size`         | Number of people in the household                      |
| `Avg_Temperature_C`      | Average temperature in Celsius                         |
| `Has_AC`                 | Indicates whether the household has an air conditioner |
| `Peak_Hours_Usage_kWh`   | Energy consumed during peak hours                      |

---

## 🛠️ Technologies Used

* Python
* Pandas
* Matplotlib
* Scikit-learn
* Jupyter Notebook / Google Colab

---

## 🔄 Machine Learning Workflow

1. Load the dataset using Pandas
2. Explore the dataset using:

   * `head()`
   * `info()`
   * `describe()`
   * `isnull().sum()`
3. Remove missing values
4. Select relevant features
5. Split the data into training and testing sets
6. Apply Polynomial Features with degree 2
7. Train a Linear Regression model
8. Generate predictions
9. Evaluate the model using:

   * MAE
   * MSE
   * RMSE
   * R² Score
10. Compare actual and predicted energy consumption
11. Visualize the results using a scatter plot

---

## 🤖 Model

### Polynomial Regression

Polynomial Regression is used to model nonlinear relationships between the independent variables and the target variable.

The model uses:

```python
PolynomialFeatures(degree=2)
```

The data is split into:

* **80% Training Data**
* **20% Testing Data**

---

## 📊 Model Performance

The trained Polynomial Regression model achieved the following results on the test dataset:

| Metric   | Result |
| -------- | -----: |
| MAE      | 0.5763 |
| MSE      | 0.5393 |
| RMSE     | 0.7344 |
| R² Score | 0.9817 |

### Metric Explanation

* **MAE (Mean Absolute Error):** Average absolute difference between actual and predicted values.
* **MSE (Mean Squared Error):** Average squared difference between actual and predicted values.
* **RMSE (Root Mean Squared Error):** Square root of MSE, expressed in the target variable's units.
* **R² Score:** Measures how much of the variation in energy consumption is explained by the model.

---

## 📈 Actual vs Predicted Values

Example predictions from the test dataset:

| Actual Energy | Predicted Energy |
| ------------: | ---------------: |
|           7.2 |            7.255 |
|          18.4 |           18.450 |
|          11.3 |           11.433 |
|          12.5 |           12.520 |
|          20.0 |           20.898 |
|          15.9 |           15.173 |
|           2.7 |            2.740 |
|          16.9 |           18.075 |
|           2.5 |            2.783 |
|           8.7 |            9.126 |

The scatter plot compares the **actual energy consumption** with the **predicted energy consumption**.

---

## 💻 Code

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures

# Load dataset
df = pd.read_csv(
    "/content/household_energy_consumption - household_energy_consumption.csv"
)

# Remove missing values
df = df.dropna()

# Select features and target
X = df[
    [
        "Household_Size",
        "Avg_Temperature_C",
        "Peak_Hours_Usage_kWh"
    ]
]

y = df["Energy_Consumption_kWh"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Polynomial features
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Prediction
y_pred = model.predict(X_test_poly)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("Polynomial Regression Results")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)

# Actual vs Predicted
result = pd.DataFrame({
    "Actual Energy": y_test.values,
    "Predicted Energy": y_pred
})

print("\nActual vs Predicted Energy Consumption")
print(result.head(10))

# Visualization
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Energy Consumption")
plt.ylabel("Predicted Energy Consumption")
plt.title("Actual vs Predicted Energy Consumption")
plt.show()
```

---

## 📁 Project Structure

```text
Household-Energy-Consumption-Prediction/
│
├── household_energy_consumption.csv
├── household_energy_prediction.ipynb
├── README.md
└── requirements.txt
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/household-energy-consumption-prediction.git
```

### 2. Install dependencies

```bash
pip install pandas matplotlib scikit-learn
```

### 3. Run the Jupyter Notebook

```bash
jupyter notebook
```

Open the project notebook and run the cells.

---

## 🔮 Future Improvements

* Include `Has_AC` as a model feature
* Convert `Date` into useful time-based features
* Compare Polynomial Regression with Random Forest and Gradient Boosting
* Perform feature correlation analysis
* Add hyperparameter tuning
* Create an interactive energy prediction dashboard
* Deploy the model as a web application

---

## 👨‍💻 Author

** Thirupathi Rajan B **

BCA Student | Python | Machine Learning | Web Development

📍 Thoothukudi, India

---

## ⭐ Conclusion

This project demonstrates how **Polynomial Regression** can be used to predict household energy consumption based on household and environmental characteristics. The model is evaluated using standard regression metrics and visualized through an actual-vs-predicted plot.
