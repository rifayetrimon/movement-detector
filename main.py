from fastapi import FastAPI, File, UploadFile
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import io
import torch
import uvicorn
from camera_frame import capture_frame_from_camera  # import your function

app = FastAPI()

processor = AutoImageProcessor.from_pretrained("DunnBC22/vit-base-patch16-224-in21k_Human_Activity_Recognition")
model = AutoModelForImageClassification.from_pretrained("DunnBC22/vit-base-patch16-224-in21k_Human_Activity_Recognition")

# Your existing /predict endpoint remains unchanged

@app.post("/predict_from_camera")
async def predict_from_camera():
    camera_url = "rtsp://admin:JZRGJS@192.168.0.104:554/h264/ch01/sub/av_stream?tcp" # your camera URL here

    image_bytes = capture_frame_from_camera(camera_url)
    if image_bytes is None:
        return {"error": "Failed to capture image from camera"}

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    return {
        "predicted_label": model.config.id2label[outputs.logits.argmax(-1).item()],
        "predicted_probability": torch.softmax(outputs.logits, dim=1).max().item()
    }




# from fastapi import FastAPI, File, UploadFile
# from transformers import AutoImageProcessor, AutoModelForImageClassification
# from PIL import Image
# import io
# import torch

# app = FastAPI()

# processor = AutoImageProcessor.from_pretrained("DunnBC22/vit-base-patch16-224-in21k_Human_Activity_Recognition")
# model = AutoModelForImageClassification.from_pretrained("DunnBC22/vit-base-patch16-224-in21k_Human_Activity_Recognition")
# # print(model.config.id2label)
 
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     # Read image bytes from uploaded file
#     image_bytes = await file.read()
#     image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

#     # Preprocess image to tensor inputs
#     inputs = processor(images=image, return_tensors="pt")

#     # Run model inference without gradient calculation
#     with torch.no_grad():
#         outputs = model(**inputs)

#     return {
#         "predicted_label": model.config.id2label[outputs.logits.argmax(-1).item()],
#         "predicted_probability": torch.softmax(outputs.logits, dim=1).max().item()
#         }



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)