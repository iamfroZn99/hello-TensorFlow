import tensorflow as tf

data = tf.keras.datasets.mnist
(training_images, training_labels),(test_images, test_labels) = data.load_data()

training_images = training_images/255
test_images = test_images/255

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),tf.keras.layers.Dense(128, activation = tf.nn.relu), tf.keras.layers.Dense(10, activation = tf.nn.softmax)])

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(training_images, training_labels, epochs = 50)
