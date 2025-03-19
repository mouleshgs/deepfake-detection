import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

def create_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),  
        Dropout(0.5), 
        layers.Dense(512, activation='relu'),  
        Dropout(0.5), 
        layers.Dense(1, activation='sigmoid')  
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_save_model(epochs=7):
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        horizontal_flip=True,
        rotation_range=20,
        zoom_range=0.2,
        shear_range=0.2,
        brightness_range=[0.8, 1.2]
    )

    train_gen = datagen.flow_from_directory(
        "./dataset",
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        "./dataset",
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    model = create_model()

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=2, verbose=1
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        './output/best_model.keras', monitor='val_loss', save_best_only=True, verbose=1
    )

    model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=[early_stopping, lr_scheduler, checkpoint]
    )

    os.makedirs("./output", exist_ok=True)
    model.save("./output/deepfake_detector_vgg16.h5")
    print("Model saved to ./output/deepfake_detector_vgg16.h5")


if __name__ == "__main__":
    train_and_save_model()
