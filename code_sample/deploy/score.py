import os
import pickle
import json
import numpy as np
import requests
import wget
from tensorflow import keras as keras
from tensorflow.keras.preprocessing import image


def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'mobilenet_dog.h5')
    # deserialize the model file back into a sklearn model
    model = keras.models.load_model(model_path)    

def load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)                    
    img_tensor = np.expand_dims(img_tensor, axis=0)         
    img_tensor /= 255. 
    return img_tensor

def run(input_data):
    try:
        data = json.loads(input_data)["data"]
        img_path = wget.download(data)
        image_tensor = load_image(img_path)
        result = model.predict(image_tensor)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        # return error message back to the client
        return json.dumps({"error": result})
