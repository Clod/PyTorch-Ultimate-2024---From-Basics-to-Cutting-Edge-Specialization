#%% 
from ultralytics import YOLO
import torch

# sources: 
# https://docs.ultralytics.com/cli/
# https://docs.ultralytics.com/cfg/
# %% load the model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("yolov8n.pt") 

model = YOLO("yolo11n.yaml")  # build a new model from scratch
model = YOLO("yolo11n.pt") 

 # load a pretrained model (recommended for training)
# %% Train the model

# Check if MPS is available
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
device='mps'
print(f"Using device: {device}")

results = model.train(data="train_custom/masks.yaml", epochs=1, imgsz=512, batch=4, verbose=True, device=device)
# device=0...GPU
# %% Export the model
model.export()
# %% 
import torch
# %%
torch.cuda.is_available()
# %%
