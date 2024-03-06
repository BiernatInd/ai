import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Wczytanie zapisanego modelu
model = load_model('model.h5')

# Ścieżka do folderu zawierającego dane testowe
test_dir = 'valid'

# Lista klas
class_names = ['bmw-i7', 'bmw-i8', 'bmw-m2', 'bmw-m3', 'bmw-x5']

# Przechowywanie prawidłowych etykiet i przewidywanych etykiet
true_labels = []
predicted_labels = []

# Przechodzenie przez dane testowe
for class_name in class_names:
    class_dir = os.path.join(test_dir, class_name)
    for image_path in os.listdir(class_dir):
        img = image.load_img(os.path.join(class_dir, image_path), target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalizacja
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        predicted_labels.append(predicted_class)
        true_labels.append(class_names.index(class_name))

# Obliczanie dokładności
accuracy = np.mean(np.array(predicted_labels) == np.array(true_labels))
print("Dokładność modelu:", accuracy)
