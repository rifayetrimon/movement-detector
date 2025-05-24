from fastapi import FastAPI, File, UploadFile
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import io
import torch


app = FastAPI()

