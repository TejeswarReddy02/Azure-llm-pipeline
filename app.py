from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


# 1. Initialize the Web App
app = FastAPI(title="LLM Workload Predictor API")

# 2. Load the trained model into memory when the server starts
try:
    with open('models/llm_workload_predictor.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None

# 3. Define what the incoming web data should look like
class PredictionRequest(BaseModel):
    ContextTokens: int
    WorkloadType: int  # 0 for Chat, 1 for Code
    Hour: int
    
    
@app.get("/")
def read_root():
    return FileResponse('static/index.html')

# 4. Create the API Endpoint
@app.post("/predict")
def predict_workload(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model file not found. Train the model first!")
    
    # Format the data for the model
    input_data = pd.DataFrame([{
        'ContextTokens': request.ContextTokens,
        'WorkloadType': request.WorkloadType,
        'Hour': request.Hour
    }])
    
    # Generate prediction
    prediction = model.predict(input_data)[0]
    
    # Determine the Server Routing Logic
    routing = "Compute-Optimized (High Prefill)" if request.WorkloadType == 1 else "Memory-Optimized (High Decode)"
    
    return {
        "predicted_generated_tokens": int(prediction),
        "recommended_routing": routing
    }
    
app.mount("/", StaticFiles(directory="static"), name="static")