import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def extract_features(image_path):
    img_preprocessed = load_and_preprocess_image(image_path)
    features = base_model.predict(img_preprocessed)
    return features.flatten()  # Flatten to a 1D array
