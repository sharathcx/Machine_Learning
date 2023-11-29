import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from gradient_descent import GradientDecent
import numpy as np
from fastapi.responses import FileResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*']
)
obj1 = GradientDecent()


@app.post("/upload")
async def upload_file(file: UploadFile = File()):
    global obj1
    path = "C:/Users/shara/OneDrive/Desktop/Machine Learning/data.xlsx"
    with open(path, "wb") as data:
        data.write(file.file.read())

    df = pd.read_excel(path)
    x_train = np.array(df.iloc[:, 1].values)
    y_trian = np.array(df.iloc[:, 2].values)
    obj1.x_train = x_train
    obj1.y_train = y_trian
    obj1.plotter(path="C:/Users/shara/OneDrive/Desktop/Machine Learning/plot")
    return obj1.gradient_descent()


@app.get("/get_file")
async def get_file():
    path = "C:/Users/shara/OneDrive/Desktop/Machine Learning/plot.png"
    return FileResponse(path, filename="plot.png")


@app.get("/predict")
async def predict(x: float):
    x_mean = obj1.x_train.mean()
    w, b, = obj1.gradient_descent()
    print(obj1.x_train)
    return obj1.predict(w, b, x)

