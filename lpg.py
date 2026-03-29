import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample dataset (Year, Month, Population, LPG_Tonnes)
data = {
    "Year": [2022, 2022, 2023, 2023, 2024, 2024],
    "Month": [1, 2, 1, 2, 1, 2],
    "Population": [1000, 1020, 1040, 1060, 1080, 1100],
    "LPG_Tonnes": [50, 55, 60, 65, 70, 75]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features (X) and Target (y)
X = df[["Year", "Month", "Population"]]
y = df["LPG_Tonnes"]

# Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Predict demand for future months
future_data = pd.DataFrame({
    "Year": [2025, 2025],
    "Month": [1, 2],
    "Population": [1120, 1140]
})
predictions = model.predict(future_data)

# Show predictions
print("Predicted LPG Demand (Tonnes):")
for i, val in enumerate(predictions):
    print(f"Month {future_data['Month'][i]}: {val:.2f}")

# Visualization
plt.scatter(df["Population"], y, color="blue", label="Actual")
plt.scatter(future_data["Population"], predictions, color="red", label="Predicted")
plt.xlabel("Population")
plt.ylabel("LPG Demand (Tonnes)")
plt.title("LPG Demand Prediction")
plt.legend()
plt.show()
