
from typing import Any
import mlflow
import pandas as pd

from fastapi import APIRouter, Depends, HTTPException
from requests import Session


from app.models import PredictInputData

router = APIRouter()

model_uri = "D:/full-stack-fastapi-template-master/backend/mlruns/744686476180347654/cfda36f39553454a8bf37cec34873fed/artifacts/CreditScore_Final"
model = mlflow.pyfunc.load_model(model_uri)

# @router.post("/predict", response_model=InputData)
# def predict(data: InputData):
#     # Convert input data to DataFrame
#     input_df = pd.DataFrame([data.dict()])
    
#     try:
#         # Make prediction
#         prediction = model.predict(input_df)
#         return {"prediction": prediction[0]}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    

# Define the prediction endpoint
@router.post("/predict", response_model=dict)
def predict(
    data: PredictInputData, 
    session: Session = Depends(),  # Include session if needed for database operations
    # current_user: CurrentUser = Depends(),  # Include user dependency if needed
):
    """
    Make a prediction using the ML model.
    """
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    try:
        # Make prediction
        prediction = model.predict(input_df)
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
