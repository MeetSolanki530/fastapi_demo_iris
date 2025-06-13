from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse,HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

feature_map = {'Iris-setosa' : 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2}

with open('Pipelinemodel.pkl','rb') as file:
    model = pickle.load(file)


class IrisFeatures(BaseModel):
    SepalLengthCm: float
    SepalWidthCm: float
    PetalLengthCm: float
    PetalWidthCm: float

def predict_species(input_data):
    
    input_array = pd.DataFrame([input_data],columns=[
        'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'
    ])

    encoded_prediction = model.predict(input_array)[0]

    species_name = {v : k for k,v in feature_map.items()}

    return species_name[encoded_prediction]

# # Example input (lengths and widths in cm)
# test_input = [5.1, 3.5, 1.4, 0.2]

# result = predict_species(test_input)
# print(f"Predicted Species: {result}")

@app.post('/predict')
def predict_category(input_data : IrisFeatures):

    #Convert Pydantic model to list for prediction
    input_list = [
        input_data.SepalLengthCm,
        input_data.SepalWidthCm,
        input_data.PetalLengthCm,
        input_data.PetalWidthCm
    ]


    output = predict_species(input_data=input_list)
    
    return JSONResponse(status_code=200,content={'response' : f'Predicted Category For Model Is :- {output}'})

# Templating engine setup
templates = Jinja2Templates(directory="templates")

@app.get('/',response_class=HTMLResponse)
def home(request : Request):
    return templates.TemplateResponse("index.html",{"request" : request})