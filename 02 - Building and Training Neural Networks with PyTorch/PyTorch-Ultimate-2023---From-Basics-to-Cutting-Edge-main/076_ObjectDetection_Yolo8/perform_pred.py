#%%
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# %% Load a model
# model = YOLO("yolov8n.pt")  # load our custom trained model
model = YOLO("runs/detect/train12/weights/best.pt")

# %%
result = model("test_cli/chinos.png")

# %% Display the result
img = Image.open("test_cli/chinos.png")
plt.figure(figsize=(12, 8))
plt.imshow(np.array(img))

# Plot the bounding boxes
for box in result[0].boxes.xyxy:
    x1, y1, x2, y2 = box
    plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2))

# Add labels
for box, cls in zip(result[0].boxes.xyxy, result[0].boxes.cls):
    x1, y1, _, _ = box
    label = result[0].names[int(cls)]
    plt.text(x1, y1, label, color='red', fontweight='bold')

plt.axis('off')
plt.show()

# %% command line run
# Standard Yolo

#!yolo detect predict model=yolov8n.pt source="test/kiki.jpg" conf=0.3 
# %% Masks 
#!yolo detect predict model=train_custom/masks.pt source="train_custom/test/images/IMG_0742.MOV" conf=0.3 

# %%
# Este de abajo anda
#yolo detect predict model="runs/detect/trainN/weights/best.pt" source=test_cli/maksssksksss10.png conf=0.3
