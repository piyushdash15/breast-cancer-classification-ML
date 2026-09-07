# Breast Cancer Classification Using Machine Learning

This project uses machine learning to classify breast cancer tumors as Benign (B) or Malignant (M). It is built using the Breast Cancer Wisconsin Diagnostic dataset and a Logistic Regression model.

> **Disclaimer:** This project is for educational and demonstration purposes only. It is not a medical diagnostic tool and should not be used for clinical decisions.

## 📌 Project Overview

The project covers:

-Loading and cleaning the dataset
-Exploring the data and understanding its features
-Selecting the required features
-Splitting the data into training and testing sets
-Scaling the features using StandardScaler
-Training a Logistic Regression model
-Evaluating the model's performance
-Saving the trained machine learning pipeline
-Deploying the model through a Streamlit web application

## 🧠 Machine Learning Model

For the classification task, we used Logistic Regression, a simple and effective algorithm for binary classification.

| Value | Meaning |
|---|---|
| `0` | Benign |
| `1` | Malignant |

The model works with **30 numerical features** based on measurements such as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

## 📊 Dataset

We use the **Breast Cancer Wisconsin Diagnostic dataset** for our training purposes.
This Dataset contained -
- **569 samples**
- **30 numerical input features**
- **357 Benign samples**
- **212 Malignant samples**

Before training the `id` and empty `Unnamed: 32` columns were removed as they were unnecessary for training.

## 🔧 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Google Colab / Jupyter Notebook

## ⚙️ Project Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Target Encoding
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
StandardScaler
   ↓
Logistic Regression
   ↓
Model Evaluation
   ↓
Save ML Pipeline
   ↓
Streamlit Web Application
```

## 📈 Model Performance

The model was evaluated on a test set of **114 samples**.

### Confusion Matrix

```text
                 Predicted
                B       M
Actual B       71       1
Actual M        3      39
```

These results are based on the project's test split and should not be interpreted as clinical validation.



## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd breast-cancer-ml
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

## 💾 Saving the Model

The trained preprocessing and classification pipeline can be saved using Joblib:

```python
import joblib

joblib.dump(pipeline, "breast_cancer_model.pkl")
```

The complete pipeline is saved so that the same feature scaling used during training is applied during prediction.

## 🌐 Deployment

The Streamlit application can be deployed using **Streamlit Community Cloud**.

```text
GitHub Repository
       ↓
Connect Repository
       ↓
Select app.py
       ↓
Install requirements.txt
       ↓
Deploy
       ↓
Web Application
```
**Link of the app:-** https://breast-cancer-classification-ml-2ksegvgrynw2wevyagn9se.streamlit.app/
## 🔮 Future Improvements

- Compare Logistic Regression with Random Forest, SVM, KNN and Gradient Boosting
- Hyperparameter tuning
- Cross-validation
- Feature selection
- Probability calibration
- Model explainability using SHAP or LIME
- Improved UI/UX
- REST API using FastAPI
- Model monitoring
- Validation on independent datasets

## 👨‍💻 Author

**1)Shwetank Vaibhav**\
**2)Piyush Kumar Dash**\
**3)Debashish Kumar Sahoo**\
**4)Ayush Shukla**\
**5)Pranshu Gupta**\
**6)Manvendra**
  
## ⭐ Acknowledgement
This project was created as a learning project to understand the complete machine learning workflow — from preparing and exploring a dataset to training and evaluating a model, saving the trained pipeline, and deploying it as a web application.

The main focus was to gain practical experience with the different steps involved in building and deploying a machine learning project.
