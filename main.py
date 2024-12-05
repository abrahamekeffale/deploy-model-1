from fastapi import FastAPI, Form
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Load the saved Decision Tree model
model = joblib.load("Notebooks/best_decision_tree_model.pkl")

# Define input data schema
class EmployeeData(BaseModel):
    Sex: str
    Label: str
    Status: str
    College: str
    Fild_of_Study: str
    Salary: float
    age: int
    year_of_service: int

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Employee Turnover Prediction API!"}

# Predict endpoint for JSON input
@app.post("/predict/")
def predict(data: EmployeeData):
    # Convert input data to DataFrame
    input_data = pd.DataFrame([data.dict()])

    # Ensure column order matches training data
    input_data = input_data[[
        "Sex", "Label", "Status", "College", "Fild_of_Study",
        "Salary", "age", "year_of_service"
    ]]

    # Make predictions
    prediction = model.predict(input_data)[0]

    # Decode prediction to original label (if encoded)
    prediction_label = "Retained" if prediction == 0 else "Left"

    return {"prediction": prediction_label}

# Interactive prediction endpoint
@app.post("/interactive-predict/")
def interactive_predict(
    Sex: str = Form(...),
    Label: str = Form(...),
    Status: str = Form(...),
    College: str = Form(...),
    Fild_of_Study: str = Form(...),
    Salary: float = Form(...),
    age: int = Form(...),
    year_of_service: int = Form(...)
):
    # Prepare input data
    input_data = pd.DataFrame([{
        "Sex": Sex,
        "Label": Label,
        "Status": Status,
        "College": College,
        "Fild_of_Study": Fild_of_Study,
        "Salary": Salary,
        "age": age,
        "year_of_service": year_of_service
    }])

    # Make predictions
    prediction = model.predict(input_data)[0]

    # Decode prediction to original label (if encoded)
    prediction_label = "Retained" if prediction == 0 else "Left"

    return {"prediction": prediction_label}
