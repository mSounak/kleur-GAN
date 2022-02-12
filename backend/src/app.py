from fastapi import FastAPI, File, UploadFile
import torch
import torch.optim as optim
import uvicorn
from PIL import Image
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import config
from utils import load_checkpoint, lab_to_rgb, image2lab
from generator_model import Generator


app = FastAPI(root_path='/')

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
gen = Generator()
# Optimizer of the model
optim = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
# loading the weights
load_checkpoint('/backend/models/checkpoints/gen.pth.tar', gen, optim, lr=config.LEARNING_RATE)

@app.get("/")
def read_root():
    return {"message": "Welcome to the API!"}


@app.get("/status")
def healthz():
    """
    Status check
    """
    return {"status": "ok"}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    """
    Predict the color of the image
    """

    L, ab = image2lab(file.file)
    L = L.unsqueeze(0)
    ab = ab.unsqueeze(0)

    # Predict the color
    with torch.no_grad():
        gen.eval()
        fake_colors = gen(L)
    
        # Lab to RGB
        rgb_imgs = lab_to_rgb(L, fake_colors)

        fake_img = np.squeeze(rgb_imgs).tolist()
    
    return {"image": fake_img}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)