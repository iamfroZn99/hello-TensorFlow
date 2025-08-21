import tensorflow as tf

# Load the MNIST dataset
data = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = data.load_data()

# Normalize the image data
training_images = training_images / 255.0
test_images = test_images / 255.0

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(training_images, training_labels, epochs=50)
