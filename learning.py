import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Definiowanie generatora danych dla treningowych i walidacyjnych
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

# Wczytanie danych treningowych
train_generator = train_datagen.flow_from_directory(
        'train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        classes=['bmw-i7', 'bmw-i8', 'bmw-m2', 'bmw-m3', 'bmw-x5'])

# Wczytanie danych walidacyjnych
validation_generator = valid_datagen.flow_from_directory(
        'valid',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        classes=['bmw-i7', 'bmw-i8', 'bmw-m2', 'bmw-m3', 'bmw-x5'])

# Definiowanie modelu
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')  # zmiana na 5 neuronów
])

# Kompilacja modelu
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Trenowanie modelu
model.fit(
        train_generator,
        steps_per_epoch=3,
        epochs=10,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=3)  # liczba kroków walidacji

# Zapisz wytrenowany model
model.save('model.h5')
