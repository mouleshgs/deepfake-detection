import tensorflow as tf
from sklearn.metrics import classification_report
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = tf.keras.models.load_model('./output/deepfake_detector.h5')


test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_dir = './dataset/test' 

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    shuffle=False  
)

predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype("int32")  

true_classes = test_generator.classes

report = classification_report(true_classes, predicted_classes, target_names=["Real", "Deepfake"])

print(report)

accuracy = np.sum(predicted_classes == true_classes) / len(true_classes)
print(f"Accuracy: {accuracy * 100:.2f}%")
