# Cardiovascular Disease Risk Analysis

## Overview
This project conducts a cross-sectional study to analyze health indicators associated with cardiovascular disease (CVD) using a health screening dataset. The aim is to identify significant risk factors (e.g., age, gender, blood pressure, BMI, cholesterol, smoking, alcohol consumption, physical activity) and develop predictive machine learning models to accurately predict CVD presence.

## Features
- **Data Analysis**: Explores relationships between health indicators and CVD using statistical methods (e.g., Chi-square test, Spearman's rank correlation).
- **Data Cleaning**: Handles outliers using central tendencies, binning, and robust statistical methods.
- **Predictive Modeling**: Implements machine learning models including Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine (SVM).
- **Visualization**: Includes box plots, scatterplots, and heatmaps to visualize outliers and data distributions.

## Dataset
The project uses the `Health Screening Data.csv` dataset, which contains 69,960 entries and 18 columns, including:
- Numerical features: age, height, weight, systolic BP, diastolic BP, BMI, etc.
- Categorical features: gender, cholesterol, glucose, smoking, alcohol consumption, physical activity, CVD status.
- Derived features: BMI category, age group.

The dataset has no missing values, and outliers were addressed using median imputation and IQR-based filtering.

## Requirements
To run the project, ensure you have the following Python libraries installed:
```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
```

Install dependencies using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/username/repository.git
   ```
2. Navigate to the project directory:
   ```bash
   cd repository
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Place the `Health Screening Data.csv` dataset in the project directory.
2. Run the main analysis script (e.g., `analysis.py`):
   ```bash
   python analysis.py
   ```
3. The script performs:
   - Data loading and cleaning.
   - Exploratory data analysis (EDA) with visualizations.
   - Statistical tests (e.g., Chi-square, Spearman’s correlation).
   - Training and evaluation of machine learning models (Logistic Regression, Decision Tree, Random Forest, SVM).
4. Results include model performance metrics (accuracy, precision, recall, F1 score) and visualizations.

Example code snippet for model training:
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load and preprocess data
df = pd.read_csv("Health Screening Data.csv")
X = df.drop(['id', 'cardio'], axis=1)
y = df['cardio']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM model
model = SVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
```

## Project Structure
- `Health Screening Data.csv`: Dataset used for analysis.
- `analysis.py`: Main script for data cleaning, EDA, statistical tests, and model training.
- `visualizations/`: Directory containing generated plots (e.g., box plots, scatterplots, heatmaps).
- `requirements.txt`: List of required Python libraries.

## Methodology
1. **Data Import**: Loaded the dataset using pandas.
2. **Data Cleaning**: Renamed columns for clarity, handled outliers using median imputation and IQR-based filtering.
3. **Exploratory Data Analysis**: Visualized data distributions and outliers using box plots and scatterplots.
4. **Statistical Analysis**: Conducted Chi-square tests for categorical variables and Spearman’s correlation for continuous variables.
5. **Predictive Modeling**: Trained and evaluated Logistic Regression, Decision Tree, Random Forest, and SVM models.
6. **Evaluation**: Assessed models using accuracy, precision, recall, and F1 score.

## Research Questions
1. What are the significant health indicators associated with CVD?
2. How do health indicators relate to the presence of CVD?
3. Can predictive machine learning models accurately predict CVD based on these indicators?

## Hypothesis
- **Null Hypothesis (H0)**: No significant relationship exists between health parameters and CVD risk.
- **Alternative Hypothesis (H1)**: A significant relationship exists between health parameters and CVD risk.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or feedback, contact the team:
- Deepishka Pemmasani
- Poorya Reddy Vanga
- Surya Tejaswi Mallidi
- Raaijtha Muthyala
- Daniel Adepoju
- Surja Tejaswi Mallidi

Affiliation: Department of Biohealth Informatics, IUPUI
```
