import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
import cv2
import numpy as np

def preprocess_image(img):
    img = img.astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    return preprocess_input(img)

# 1. Load your saved brain
model = tf.keras.models.load_model("model_binary.keras")

# 2. Point to your data
datagen = ImageDataGenerator(preprocessing_function=preprocess_image)
test_data = datagen.flow_from_directory(
    'dataset/train', # Or 'dataset/test' if you have one
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# 3. Calculate the score
loss, accuracy = model.evaluate(test_data)
print(f"\n✅ Your Project Accuracy is: {accuracy * 100:.2f}%")