# Sample dataset (Population vs LPG_Tonnes)
data = [
    (1000, 50),
    (1020, 55),
    (1040, 60),
    (1060, 65),
    (1080, 70),
    (1100, 75)
]

# Calculate simple linear regression manually
n = len(data)
sum_x = sum(p for p, _ in data)
sum_y = sum(d for _, d in data)
sum_xy = sum(p*d for p, d in data)
sum_x2 = sum(p*p for p, _ in data)

# Regression coefficients: y = a + b*x
b = (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x**2)
a = (sum_y - b*sum_x) / n

print("Regression Equation: LPG_Tonnes = {:.2f} + {:.4f}*Population".format(a, b))

# Predict demand for future populations
future_populations = [1120, 1140]
for pop in future_populations:
    prediction = a + b*pop
    print(f"Predicted LPG demand for population {pop}: {prediction:.2f} tonnes")
