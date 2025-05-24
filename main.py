# from fastapi import FastAPI, File, UploadFile
# from transformers import AutoImageProcessor, AutoModelForImageClassification
# from PIL import Image
# import io
# import torch

# app = FastAPI()

# # Load processor and model once at startup
# image_processor = AutoImageProcessor.from_pretrained("DunnBC22/vit-base-patch16-224-in21k_Human_Activity_Recognition")
# model = AutoModelForImageClassification.from_pretrained("DunnBC22/vit-base-patch16-224-in21k_Human_Activity_Recognition")

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     # Read image bytes from uploaded file
#     image_bytes = await file.read()
#     image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

#     # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#     # image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
#     # Preprocess the image
#     inputs = image_processor(images=image, return_tensors="pt")

#     # Run model inference
#     with torch.no_grad():
#         outputs = model(**inputs)

#     # Process the output: Get the predicted class and confidence
#     logits = outputs.logits  # Shape: [batch_size, num_classes]
#     predicted_class = torch.argmax(logits, dim=1).item()  # Get the index of the highest score
#     probabilities = torch.softmax(logits, dim=1)  # Convert logits to probabilities
#     predicted_probability = probabilities[0, predicted_class].item()  # Probability of the predicted class

#     # Assuming the model has a label mapping; access it via model.config.id2label
#     if hasattr(model.config, 'id2label'):
#         predicted_label = model.config.id2label.get(predicted_class, "Unknown Label")
#     else:
#         predicted_label = f"Class {predicted_class}"  # Fallback if no label mapping

#     # # Print the results
#     # print(f"Predicted Label: {predicted_label}")
#     # print(f"Predicted Probability: {predicted_probability:.4f}")

#     return {
#         "predicted_label": predicted_label, 
#         "predicted_probability": predicted_probability
#         }

#     # Example output might look like: Predicted Label: some_activity, Predicted Probability: 0.8500


from fastapi import FastAPI, File, UploadFile
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import io
import torch

app = FastAPI()

processor = AutoImageProcessor.from_pretrained("DunnBC22/vit-base-patch16-224-in21k_Human_Activity_Recognition")
model = AutoModelForImageClassification.from_pretrained("DunnBC22/vit-base-patch16-224-in21k_Human_Activity_Recognition")
# print(model.config.id2label)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image bytes from uploaded file
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess image to tensor inputs
    inputs = processor(images=image, return_tensors="pt")

    # Run model inference without gradient calculation
    with torch.no_grad():
        outputs = model(**inputs)

    return {
        "predicted_label": model.config.id2label[outputs.logits.argmax(-1).item()],
        "predicted_probability": torch.softmax(outputs.logits, dim=1).max().item()
        }
