import os
import pickle

def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def save_model(model, model_path):
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
