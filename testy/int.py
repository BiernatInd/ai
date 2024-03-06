import unittest
import os
import tensorflow as tf
from test import predict_car_brand  # Zastąp 'your_script' nazwą Twojego skryptu zawierającego funkcję predict_car_brand

class TestCarBrandPrediction(unittest.TestCase):
    def setUp(self):
        self.model = tf.keras.models.load_model('model.h5')  # Wczytaj wytrenowany model przed każdym testem
        self.class_labels = {0: 'bmw-i7', 1: 'bmw-i8', 2: 'bmw-m2', 3: 'bmw-m3', 4: 'bmw-x5'}

    def test_car_brand_prediction(self):
        test_images = ['train/bmw-i7/1.png', 'train/bmw-i8/1.png', 'train/bmw-m2/1.png', 'train/bmw-m3/1.png', 'train/bmw-x5/1.png']
        
        for image_path in test_images:
            predicted_brand = predict_car_brand(image_path, self.model, self.class_labels)
            actual_brand = os.path.basename(os.path.dirname(image_path))  # Pobierz rzeczywistą markę samochodu z ścieżki
            self.assertEqual(predicted_brand, actual_brand)

if __name__ == '__main__':
    unittest.main()
