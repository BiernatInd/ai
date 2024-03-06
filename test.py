import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

def predict_car_brand(image_path, model, class_labels):
    img = image.load_img(image_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    
    # Mapowanie etykiet do nazw klas
    predicted_label = class_labels[np.argmax(classes)]
    
    return predicted_label

# Wczytaj wytrenowany model
model = tf.keras.models.load_model('model.h5')

# Wczytaj mapowanie klas na nazwy
class_labels = {0: 'bmw-i7', 1: 'bmw-i8', 2: 'bmw-m2', 3: 'bmw-m3', 4: 'bmw-x5'}

# Ścieżka do zdjęcia samochodu
image_path = '3.jpeg'

# Przewidziana marka samochodu
predicted_brand = predict_car_brand(image_path, model, class_labels)
print("Predicted car brand:", predicted_brand)
