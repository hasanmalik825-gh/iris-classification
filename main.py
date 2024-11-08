# Import FastAPI and necessary packages
from fastapi import FastAPI,Query
import uvicorn
import logging
import pickle
import pandas as pd
# Initialize FastAPI app
app = FastAPI()

x_cols = ['sepal length (cm)',
          'sepal width (cm)',
          'petal length (cm)',
          'petal width (cm)']
with open('iris-dt.pickle', 'rb') as f:
    model = pickle.load(f)

def predict_class(s_length, s_width, p_length, p_width):
    return model.predict(pd.DataFrame([[float(i) for i in [s_length, s_width, p_length, p_width]]], columns=x_cols))[0]

@app.get("/")
async def index() -> dict:
    """
    Index route to check if the API is working.
    Returns:
        dict: The status of the API.
    """
    logging.info("Index is working")
    return {
        "Status": True,
        "message": "API is working.."
    }

@app.post("/predict")
async def welcome_message(
    s_length: str = Query(..., description="sepal length (cm)"),
    s_width: str = Query(..., description="sepal width (cm)"),
    p_length: str = Query(..., description="petal length (cm)"),
    p_width: str = Query(..., description="petal width (cm)")
) -> dict:
    
    return {
        "class":f"{predict_class(s_length, s_width, p_length, p_width)}"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
