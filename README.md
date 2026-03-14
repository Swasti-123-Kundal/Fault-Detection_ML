# Chemical Process Fault Detection using Machine Learning

## Overview
This project demonstrates a machine learning approach for detecting faults in a simulated chemical process. Sensor data such as temperature, pressure, flow rate, and concentration are used to classify whether the system is operating under normal conditions or experiencing a fault.

A Random Forest classification model is trained using Python and Scikit-Learn to identify abnormal process behavior.

---

## Features
- Synthetic chemical process dataset generation
- Fault detection using Random Forest classifier
- Data visualization and analysis
- Model evaluation using classification metrics and confusion matrix
- Feature importance analysis

---

## Technologies Used
- Python
- NumPy
- Pandas
- Scikit-Learn
- Matplotlib
- Seaborn

---

## Dataset

The dataset used in this project is **synthetically generated** to simulate sensor readings from a chemical process monitoring system.

### Variables

| Feature | Description |
|-------|-------------|
| Temperature | Reactor temperature |
| Pressure | Process pressure |
| Flow Rate | Flow rate of the system |
| Concentration | Chemical concentration level |
| Fault | Target variable (0 = Normal, 1 = Fault) |

### Fault Condition

A fault is triggered when:

- Temperature > 130  
- Pressure > 6

This simulates abnormal operating conditions in a chemical process.

---

## Project Workflow

1. Generate synthetic dataset
2. Split dataset into training and testing sets
3. Train Random Forest classification model
4. Evaluate model performance
5. Visualize results using graphs

---

## Results

The model successfully detects faults in simulated chemical process conditions using sensor data.

Visualization outputs include:
- Confusion Matrix
- Feature Importance
- Sensor Data Distributions

---

## How to Run the Project

### 1. Install required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
