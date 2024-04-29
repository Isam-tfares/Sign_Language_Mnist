from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import numpy as np
import pickle
# import joblib
from tensorflow import keras
from io import BytesIO
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Loading LabelBinarizer (lb) from the saved file
with open('C:\\Users\\hp\\Desktop\\Sign_Language_MNIST\\saved_objects\\label_binarizer.pkl', 'rb') as file:
    loaded_lb = pickle.load(file)
def get_label(y):
    return loaded_lb.inverse_transform(y)[0][0]

@csrf_exempt
def predict_image(request):
    prediction=[0,0,0]
    model_path = 'C:\\Users\\hp\\Desktop\\Sign_Language_MNIST\\model.h5'
    loaded_model = keras.models.load_model(model_path)
    if request.method == 'POST' and request.FILES.get('image'):
        # Get the uploaded image from the request
        image = request.FILES['image']
        
        # Read the image using BytesIO and decode with OpenCV
        image_data = BytesIO(image.read())
        img = cv2.imdecode(np.frombuffer(image_data.read(), np.uint8), cv2.IMREAD_COLOR)
        
        if img is not None:
            img = cv2.resize(img, (64, 64))
            img = img / 255.0
            
            # Get the dimensions of the processed image
            height, width, channels = img.shape

            img=img.reshape((1, 64, 64, 3))
            prediction = loaded_model.predict(img)
            index=np.argmax(prediction)
            y=np.zeros((1,24))
            y[0][index]=1
            lettre=get_label(y)
            result = {'prediction': lettre}
            return JsonResponse(result)
        else:
            return JsonResponse({'error': 'Failed to read the image'})
    else:
        return JsonResponse({'error': 'Image file not found or method not allowed'})



