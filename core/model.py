# core/model.py

import configparser
import streamlit as st
from ultralytics import YOLO

# read model paths from config.ini
config = configparser.ConfigParser()
config.read("config.ini")

MODEL_PATHS = {
    "Nano": config["MODELS"]["nano"],
    "Small": config["MODELS"]["small"],
}

@st.cache_resource
def load_model(model_name: str):
    model_path = MODEL_PATHS.get(model_name)

    # for unknown errors
    if model_path is None:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_PATHS.keys())}")
    
    return YOLO(model_path, verbose=True)
