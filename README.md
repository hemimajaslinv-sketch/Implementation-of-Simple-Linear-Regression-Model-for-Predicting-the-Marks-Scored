Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the dataset containing Study Hours (X) and Marks Scored (Y).
2. Calculate the mean of the study hours and marks, then compute the slope (m) and y-intercept (b) using the least squares formulas.
3. Form the linear regression equation (Y = mX + b) and use it to predict the marks for a new, given number of study hours.
4. Plot the actual dataset as a scatterplot along with the calculated regression line to visualize the model.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Hemima Jaslin V
RegisterNumber: 212224250006
*/


```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create Dataset
data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks_Scored":  [35, 40, 50, 55, 60, 65, 70, 80, 85, 95]
}
df = pd.DataFrame(data)

print("Dataset:\n", df.head())

# Split into Features and Target
X = df[["Hours_Studied"]]
y = df["Marks_Scored"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
print("\nModel Parameters:")
print("Intercept (b0):", model.intercept_)
print("Slope (b1):", model.coef_[0])

print("\nEvaluation Metrics:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Visualization
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, model.predict(X), color='red', linewidth=2, label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression: Predicting Marks")
plt.legend()
plt.grid(True)
plt.show()

# Custom Input Prediction
hours = 5
custom_input = pd.DataFrame({"Hours_Studied": [hours]})
predicted_marks = model.predict(custom_input)

print(f"\nPredicted marks for {hours} hours of study = {predicted_marks[0]:.2f}")
```



## Output:
Dataset:

<img width="193" height="338" alt="image" src="https://github.com/user-attachments/assets/d2e8dd56-c338-4ab5-b039-f06f8629fa7e" />

Model parameters:

<img width="234" height="96" alt="image" src="https://github.com/user-attachments/assets/b15bcf55-f02f-4d45-89e5-e2cd3507fbd5" />

Map plot:

<img width="532" height="372" alt="image" src="https://github.com/user-attachments/assets/23f39933-b6c4-49e3-afeb-b9a054a052f1" />

Output:

<img width="290" height="46" alt="image" src="https://github.com/user-attachments/assets/54dd4165-a19b-4844-a8a8-31cafab89b18" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
