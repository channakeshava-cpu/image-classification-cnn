import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("cnn_model.h5")

# CIFAR-10 class labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Get image path from user
img_path = input("Enter image path: ").strip()

# Load and preprocess image
img = image.load_img(img_path, target_size=(32, 32))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)
pred_class = np.argmax(predictions)

print(f"\n🔍 Predicted class: {class_names[pred_class]}")

# Optionally check if it's cat or dog
if class_names[pred_class] in ['cat', 'dog']:
    print("✅ The image contains a cat or dog.")
else:
    print("❌ The image is NOT a cat or dog.")
