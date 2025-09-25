# Titanic Survival Prediction

This repository presents a supervised machine learning pipeline to predict passenger survival on the Titanic dataset.  
The dataset originates from the well-known Kaggle Titanic competition and serves as a benchmark problem for classification tasks.

---

## Project Overview
The objective of this project is to build, train, and evaluate multiple supervised classification models to determine survival outcomes based on passenger attributes such as:

- Passenger Class (Pclass)  
- Gender (Sex)  
- Age  
- Fare  
- Embarked Port  
- Family-related features (SibSp, Parch)  

---

## Methodology
1. **Data Preprocessing**
   - Handling missing values (Age, Embarked, Cabin).  
   - Encoding categorical variables using label encoding and one-hot encoding.  
   - Outlier detection and removal (IQR method applied to Age).  

2. **Feature Engineering**
   - Selection of relevant features.  
   - Transformation of categorical features.  
   - Optional scaling of continuous features.  

3. **Model Development**
   The following supervised learning models were implemented and trained:
   - Logistic Regression  
   - K-Nearest Neighbors (KNN)  
   - Support Vector Machine (SVM)  
   - Decision Tree Classifier  
   - Random Forest Classifier  
   - Gradient Boosting Classifier  
   - XGBoost Classifier  
   - LightGBM Classifier  

4. **Model Evaluation**
   - Accuracy score comparison.  
   - Visualization of model performance.  
   - Ranking models by predictive accuracy.  

---

## Results
| Model                   | Accuracy |
|--------------------------|----------|
| Logistic Regression      | 0.80     |
| KNN                      | 0.78     |
| SVM                      | 0.79     |
| Decision Tree            | 0.76     |
| Random Forest            | 0.81     |
| Gradient Boosting        | 0.82     |
| XGBoost                  | 0.83     |
| LightGBM                 | 0.84     |

The LightGBM Classifier achieved the highest accuracy among all implemented models.

---

## Technology Stack
- **Programming Language:** Python  
- **Data Manipulation:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Machine Learning Frameworks:** Scikit-learn, XGBoost, LightGBM  

---

## How to Reproduce
1. Clone the repository:
   ```bash
   git clone https:https://github.com/Muskankumari13/Supervised-Learning-Models
