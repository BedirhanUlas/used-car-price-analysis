Used Car Price Prediction

What Drives the Price of a Car?

Analyzing the key factors that influence used car prices using machine learning models.

1. Project Overview

This project aims to analyze the factors affecting used car prices using a dataset from Kaggle. The dataset contains various attributes such as year, odometer, manufacturer, fuel type, transmission, and price.

By using data preprocessing, exploratory data analysis, and machine learning models, we build a model that predicts the price of a used car based on its attributes.

The objective is to help used car dealerships optimize their pricing strategy by understanding which factors impact car prices the most.

2. Dataset Information
	•	Source: Kaggle
	•	Number of Entries: 426,880
	•	Key Features:
	•	year: Manufacturing year of the vehicle
	•	odometer: Total mileage of the vehicle
	•	price: Selling price of the vehicle (Target Variable)
	•	manufacturer: Brand of the car
	•	fuel: Fuel type (Gas, Diesel, Electric, etc.)
	•	transmission: Type of transmission (Manual, Automatic)
	•	condition: Vehicle condition (New, Good, Fair, etc.)

3. Installation & Requirements

To run this project, you need the following Python libraries:

pip install pandas numpy matplotlib seaborn scikit-learn

Steps to execute the notebook:
	1.	Upload vehicles.csv to the main directory.
	2.	Open prompt_ll.ipynb in Google Colab or Jupyter Notebook.
	3.	Run all cells to explore the data, build models, and evaluate predictions.

4. Project Structure
📂 Used-Car-Price-Prediction
│── 📄 README.md
│── 📄 vehicles.csv  # Dataset used in the project
│── 📄 prompt_ll.ipynb  # Jupyter Notebook containing all code

5. Methodology

This project follows the CRISP-DM framework, a standard data science process:

✅ Business Understanding: Identifying key factors influencing used car prices.
✅ Data Understanding: Exploring and visualizing the dataset.
✅ Data Preparation: Handling missing values, outliers, and feature engineering.
✅ Modeling: Training a Linear Regression model for price prediction.
✅ Evaluation: Assessing model performance using MAE and MSE.
✅ Deployment: Summarizing findings and providing business recommendations.

6. Results & Key Findings
	•	The most important factors affecting price are the vehicle’s year and odometer reading.
	•	Newer cars tend to be more expensive.
	•	Higher mileage leads to lower car prices.
	•	The model’s average error (MAE) is $3,898.29, and MSE is 12,984,103.
	•	Further improvements can be made by adding more features and testing advanced models.

7. Future Work
	•	Exploring advanced models (e.g., Random Forest, XGBoost, Neural Networks).
	•	Using feature selection techniques to identify the most relevant variables.
	•	Applying data transformations (log transformation) to handle price distribution skewness.
	•	Developing a real-time price prediction API for car dealerships.

8. Contributors

👤 Bedirhan Ulas
📧 Email: bedirhanulas@outlook.com
🔗 LinkedIn: linkedin.com/in/bedirhanulas

9. License

This project is developed under Berkeley Extension and follows its academic guidelines.