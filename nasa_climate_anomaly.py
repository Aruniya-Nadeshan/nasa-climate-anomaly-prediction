import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('GLB.Ts+dSST.csv', skiprows=1)

df = df[['Year', 'J-D']].dropna()
df['J-D'] = pd.to_numeric(df['J-D'], errors='coerce')
df = df.dropna()

X = df['Year'].values.reshape(-1, 1)
y = df['J-D'].values

model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)

plt.figure(figsize=(10, 5))
plt.scatter(X, y, color='blue', label='Actual NASA Anomalies', alpha=0.5)
plt.plot(X, predictions, color='red', linewidth=2, label='ML Linear Trend')
plt.title('NASA Global Temperature Anomaly Regression')
plt.xlabel('Year')
plt.ylabel('Anomaly (°C)')
plt.legend()

plt.savefig('nasa_trends.png')
print("Success! Plot saved as 'nasa_trends.png'.")
plt.show()

