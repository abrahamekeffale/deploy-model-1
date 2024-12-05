from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model = joblib.load("Notebooks/best_decision_tree_model.pkl")

# Set up templates for rendering HTML
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Root route to display the form
@app.get("/", response_class=HTMLResponse)
def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

# Predict route to handle form submission and display prediction
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    Sex: str = Form(...),
    Label: str = Form(...),
    Status: str = Form(...),
    College: str = Form(...),
    Fild_of_Study: str = Form(...),
    Salary: float = Form(...),
    age: int = Form(...),
    year_of_service: int = Form(...)
):
    # Create a DataFrame from form inputs
    input_data = pd.DataFrame([{
        "Sex": Sex,
        "Label": Label,
        "Status": Status,
        "College": College,
        "Fild of Study": Fild_of_Study,
        "Salary": Salary,
        "age": age,
        "year_of_service": year_of_service
    }])

    # Predict the outcome
    prediction = model.predict(input_data)[0]
    prediction_label = "Retained" if prediction == 0 else "Left"

    # Render the result in HTML
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "prediction": prediction_label,
            "data": input_data.to_dict(orient="records")[0]
        }
    )
