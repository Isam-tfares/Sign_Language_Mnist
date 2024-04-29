{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "0de82ec0",
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n",
    "import numpy as np\n",
    "import pickle\n",
    "from tensorflow import keras\n",
    "import os\n",
    "os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'\n",
    "\n",
    "# Loading LabelBinarizer (lb) from the saved file\n",
    "with open('C:\\\\Users\\\\hp\\\\Desktop\\\\Sign_Language_MNIST\\\\saved_objects\\\\label_binarizer.pkl', 'rb') as file:\n",
    "    loaded_lb = pickle.load(file)\n",
    "def get_label(y):\n",
    "    return loaded_lb.inverse_transform(y)[0][0]\n",
    "\n",
    "def predict_image(img):\n",
    "    prediction=[0,0,0]\n",
    "    model_path = 'C:\\\\Users\\\\hp\\\\Desktop\\\\Sign_Language_MNIST\\\\model.h5'\n",
    "    loaded_model = keras.models.load_model(model_path)\n",
    "    if img is not None:\n",
    "        img = cv2.resize(img, (64, 64))\n",
    "        img = img / 255.0\n",
    "        img=img.reshape((1, 64, 64, 3))\n",
    "        prediction = loaded_model.predict(img)\n",
    "        index=np.argmax(prediction)\n",
    "        y=np.zeros((1,24))\n",
    "        y[0][index]=1\n",
    "        lettre=get_label(y)\n",
    "        result = {'prediction': lettre}\n",
    "        return result\n",
    "    else:\n",
    "        return \"img is None\"\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e40cc602",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
