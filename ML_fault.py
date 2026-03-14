# Chemical Process Fault Detection using Machine Learning

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Step 2: Create Synthetic Dataset
np.random.seed(42)

data_size = 1000

temperature = np.random.normal(120,10,data_size)
pressure = np.random.normal(5,1,data_size)
flow_rate = np.random.normal(30,5,data_size)
concentration = np.random.normal(0.8,0.1,data_size)

# Fault condition
fault = (temperature > 130) | (pressure > 6)

# Create dataframe
df = pd.DataFrame({
    "temperature":temperature,
    "pressure":pressure,
    "flow_rate":flow_rate,
    "concentration":concentration,
    "fault":fault.astype(int)
})

print("First 5 rows of dataset:\n")
print(df.head())


# Step 3: Define Features and Target
X = df.drop("fault",axis=1)
y = df["fault"]


# Step 4: Split Dataset
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

print("\nTraining Size:",X_train.shape)
print("Testing Size:",X_test.shape)


# Step 5: Train Model
model = RandomForestClassifier(n_estimators=100)

model.fit(X_train,y_train)

print("\nModel Training Completed")


# Step 6: Predictions
pred = model.predict(X_test)


# Step 7: Accuracy
accuracy = accuracy_score(y_test,pred)

print("\nModel Accuracy:",accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test,pred))


# Step 8: Confusion Matrix
cm = confusion_matrix(y_test,pred)

plt.figure()
sns.heatmap(cm,annot=True,fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# Step 9: Feature Importance
importance = model.feature_importances_

features = X.columns

plt.figure()
plt.bar(features,importance)
plt.title("Feature Importance")
plt.xticks(rotation=45)
plt.show()


# Step 10: Data Visualization

# Temperature distribution
plt.figure()
plt.hist(df["temperature"],bins=30)
plt.title("Temperature Distribution")
plt.xlabel("Temperature")
plt.ylabel("Frequency")
plt.show()

# Pressure distribution
plt.figure()
plt.hist(df["pressure"],bins=30)
plt.title("Pressure Distribution")
plt.xlabel("Pressure")
plt.ylabel("Frequency")
plt.show()