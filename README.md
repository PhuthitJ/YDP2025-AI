# **🏦 Bank Customer Churn Prediction (YDP 2025 AI/ML Mini Project)**
This project aims to predict whether a bank customer will stop using the bank's services (churn) based on their profile and account information. It was developed as part of the Young Developer Power (YDP) 2025 AI/ML Track Mini Project.

The project involves Exploratory Data Analysis (EDA), data preprocessing, training multiple classification models, evaluating their performance, and selecting the best model (Random Forest). An optional FastAPI application is included to serve predictions via an API.

---

# **📊 Dataset**

The project uses the `Customer-Churn-Records.csv` dataset (included in the repository). It contains various customer attributes, including:

* **Demographics:** Geography, Gender, AgeAccount 

* **Information:** CreditScore, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary

* **Loyalty/Satisfaction:** Complain, Satisfaction_Score, Card_Type, Point_Earned

* **Target Variable:** `Exited` (1 if the customer churned, 0 otherwise)

Irrelevant columns like `RowNumber`, `CustomerId`, and `Surname` were dropped during preprocessing.

---

# **Workflow**

1. **Exploratory Data Analysis (EDA):**
The `YDP2025.ipynb` notebook performs initial analysis using descriptive statistics (`describe`, `info`) and visualizations (pie charts, boxplots, heatmap) to understand data distributions, correlations, and relationships with the target variable (`Exited`).

2. **Data Preprocessing:**
* Irrelevant columns were dropped.
* Categorical features (Gender, Geography, Card_Type) were encoded using LabelEncoder and One-Hot Encoding (via pd.get_dummies initially, then incorporated into a pipeline).
* Numerical features were scaled using StandardScaler.
  
3. **Model Training & Evaluation:**
* Several classification models were trained and evaluated:
   * Logistic Regression
   * Decision Tree Classifier
   * Random Forest Classifier
   * XGBoost Classifier
* Performance was measured using Accuracy, Classification Report (Precision, Recall, F1-Score), and Confusion Matrix.
* The **Random Forest Classifier** was selected as the final model due to its high performance (Accuracy: 0.999).
* *Note:* The high accuracy is likely influenced significantly by the `Complain` feature, which has a very strong correlation with `Exited`.

4. **Pipeline & Model Saving:** A `scikit-learn` pipeline (`bank_churn_pipeline.pkl`) was created to combine preprocessing steps (scaling, encoding) and the Random Forest model. This pipeline is saved using `joblib`.

5. **API (Optional):** A simple API using `FastAPI` (`app.py`) is provided to load the saved pipeline and serve predictions on new customer data.

---

# **⚙️ Setup and Installation**

1️⃣ **Clone the repository:**

```
git clone [https://github.com/PhuthitJ/YDP2025-AI.git](https://github.com/PhuthitJ/YDP2025-AI.git)
cd YDP2025-AI
```

2️⃣ **Create a virtual environment (recommended) 🐧:**

```
python -m venv venv
source venv/bin/activate  # 🪟 On Windows use `venv\Scripts\activate`
```

3️⃣ **Install dependencies:**

```
pip install pandas scikit-learn joblib fastapi uvicorn "pydantic[email]" xgboost matplotlib seaborn jupyterlab
```

*(Note: `xgboost`, `matplotlib`, `seaborn`, `jupyterlab` are mainly for running the notebook; `fastapi`, `uvicorn`, `pydantic` are for the API.)*

---

# **Usage**

1. **Jupyter Notebook** (`YDP2025.ipynb`)
* Launch JupyterLab:
```
jupyter lab
```
* Open and run the `YDP2025.ipynb` notebook to see the EDA, model training, evaluation process, and saving of the pipeline (`.pkl` file).

2. **FastAPI Application** (`app.py`)
* **Ensure** `bank_churn_pipeline.pkl` is in the same directory as `app.py`.
* **Start the API server:**
```
uvicorn app:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

* **Endpoints**:

   * `GET /`: Welcome message.
   * `GET /health`: Health check, confirms if the model is loaded.
   * `POST /predict:` Accepts customer data (JSON) and returns the churn prediction and probability.
* **Example Prediction Request** (using `curl`):
```
curl -X POST "[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)" -H "Content-Type: application/json" -d '{
  "CreditScore": 600,
  "Geography": "France",
  "Gender": "Male",
  "Age": 40,
  "Tenure": 3,
  "Balance": 60000,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 50000,
  "Complain": 0,
  "Satisfaction_Score": 3,
  "Card_Type": "SILVER",
  "Point_Earned": 500
}'
```
* **Expected Response:**
```
{
  "prediction": "Will Stay",
  "churn_probability": 0.01 
}
```
*(Probability value is an example and will vary)*

---

# **File Structure**

├── YDP2025.ipynb             # Jupyter Notebook with EDA, training, and evaluation

├── app.py                    # FastAPI application for serving predictions

├── bank_churn_pipeline.pkl   # Saved scikit-learn pipeline (model + preprocessing)

├── Customer-Churn-Records.csv # Dataset used for training and evaluation

├── YDP_Project_AI.pdf        # Project description and requirements (in Thai)

└── README.md                 # This file

---

# **Results**
The final Random Forest model achieved very high accuracy (**99.9%**) on the test set. Analysis of feature importances showed that `Complain`, `Age`, `NumOfProducts`, and `IsActiveMember` were among the most influential factors in predicting churn for this dataset and model configuration. The strong predictive power of the `Complain` feature likely contributes significantly to the high accuracy.
