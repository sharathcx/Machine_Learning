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


@app.post("/upload")
async def upload_file(file: UploadFile = File()):
    path = "C:/Users/shara/OneDrive/Desktop/Machine Learning/data.xlsx"
    with open(path, "wb") as data:
        data.write(file.file.read())

    df = pd.read_excel(path)
    x_train = np.array(df.iloc[:, 1].values)
    y_trian = np.array(df.iloc[:, 2].values)
    obj1 = GradientDecent(x_train, y_trian)
    obj1.plotter(path="C:/Users/shara/OneDrive/Desktop/Machine Learning/plot")

@app.get("/get_file")
async def get_file():
    path = "C:/Users/shara/OneDrive/Desktop/Machine Learning/plot.png"
    return FileResponse(path, filename="plot.png")

