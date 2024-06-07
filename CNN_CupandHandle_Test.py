import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img_path = '/mnt/data/Screenshot 2024-06-07 161323.png'
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Preprocess the image for the model
img_array /= 255.0

# Load the trained model (assuming you have a trained model)
model = load_model('path_to_trained_model.h5')

# Predict the pattern
prediction = model.predict(img_array)

# Assuming the model outputs a probability for the "cup and handle" pattern
if prediction[0] > 0.5:
    print("The pattern is a 'cup and handle'.")
else:
    print("The pattern is not a 'cup and handle'.")

# Display the image
plt.imshow(img)
plt.show()
