from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import nltk
import pickle
import mlflow # needed as the covidrepo pkl class is a sublcass of this
import json

app = FastAPI()
model = pickle.load(open("model.pkl","rb"))

# pd.DataFrame([["Python", "Data"]]).to_json(orient="split")
# {"columns":[0,1],"index":[0],"data":[["Python","Data"]]}
class PredictPayload(BaseModel):
    columns: list
    data: list

@app.post("/predict")
async def _predict(payload: PredictPayload):
    input = pd.read_json(payload.json(), orient='split')
    output = model.predict(None,input)
    # model.predict() returning invalid json -
    # https://stackoverflow.com/questions/38821132/bokeh-valueerror-out-of-range-float-values-are-not-json-compliant
    return json.dumps(output)
