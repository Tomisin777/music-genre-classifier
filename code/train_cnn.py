
import tensorflow as tf
from tensorflow import keras
import os

#Loading files 

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'Data', 'spectrograms')



train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(128, 128),
    batch_size=32
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(128, 128),
    batch_size=32
)

# Check it worked
#class_names = train_dataset.class_names
#print(f"Classes found: {class_names}")
#print(f"Number of training batches: {len(train_dataset)}")
#print(f"Number of validation batches: {len(val_dataset)}")

#Building CNN model 
from tensorflow.keras import layers, models
model = models.Sequential([
    #Layer 1: Find basic patterns (32 detectors)
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
     # Layer 2: Simplify
    layers.MaxPooling2D((2, 2)),
    # Layer 3: Find complex patterns (64 detectors)
    layers.Conv2D(64, (3, 3), activation='relu'),
    # Layer 4: Simplify again
    layers.MaxPooling2D((2, 2)),
    # Layer 5: Convert 2D to 1D
    layers.Flatten(),
    # Layer 6: Combine all patterns (128 neurons)
    layers.Dense(128, activation='relu'),
    # Layer 7: Final answer (10 genres)
    layers.Dense(10, activation='softmax')
])

#model.summary()

# Compile the model 

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model 

print("\n🎵 Starting training...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)

print("✅ Training complete!")

import matplotlib.pyplot as plt

# Plot training history
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

plt.tight_layout()
plt.savefig('training_history.png')
print("📊 Training plots saved!")