import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1️⃣ Load your previously trained CIFAR-10 model
model = tf.keras.models.load_model("cnn_model.h5")

# 2️⃣ Set up data generators for your real cat/dog images
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80% train / 20% validation
)

train_data = datagen.flow_from_directory(
    "real_images",
    target_size=(32, 32),  # match CIFAR-10 size
    batch_size=8,
    class_mode="sparse",
    subset="training"
)

val_data = datagen.flow_from_directory(
    "real_images",
    target_size=(32, 32),
    batch_size=8,
    class_mode="sparse",
    subset="validation"
)

# 3️⃣ Recompile with a smaller learning rate (for gentle fine-tuning)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4️⃣ Train for a few epochs
history = model.fit(train_data, validation_data=val_data, epochs=5)

# 5️⃣ Save the improved model
model.save("cnn_model_v2.h5")
print("✅ Fine-tuned model saved as cnn_model_v2.h5")
