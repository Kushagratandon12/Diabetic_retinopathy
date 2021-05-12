import tensorflow as tf
import cv2
import numpy as np
from tensorflow import keras
from keras.models import load_model

classes = ['No_Dr', 'Mild', 'Moderate', 'severe', 'Proliferative DR']


# path = 's3://mldiabeticmodeldata/diab_model.h5'
saved_model = load_model('components\model\diab_model.h5')


def decode_img(image, shape=(256, 256)):
    img = tf.image.convert_image_dtype(image, tf.float16)
    img = tf.image.resize(img, [256, 256])
    img = img.numpy().reshape(1, 256, 256, 3)
    # print(img.shape)
    return img


def model_pred(image_bytes):
    decoded2 = tf.io.decode_jpeg(image_bytes, channels=3)
    img = decode_img(decoded2)
    pred = saved_model.predict(img)
    idx = np.argmax(pred)
    # print(classes[idx])
    predictions = classes[idx]
    return predictions
