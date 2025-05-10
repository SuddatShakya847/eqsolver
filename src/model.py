import tensorflow as tf
import time
import numpy as np


class Model:
    def __init__(self, load: bool = True):
        self.class_names = ['0', '1', '2', '3', '4',
                            '5', '6', '7', '8', '9',
                            '.', '-', '+', '/', 'x']
        self.num_classes = len(self.class_names)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1.0/255, input_shape=(28, 28, 3)),
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                input_shape=(28, 28, 3),
                activation='relu',
                padding='same'
            ),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(5, 5),
                activation='relu',
            ),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(self.num_classes)
        ])
        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            metrics=['accuracy']
        )

        if load:
            self.model.load_weights("saves/model.weights.h5")

    def summary(self):
        return self.model.summary()

    def predict(self, images):
        return self.class_names[np.argmax(self.model.predict(images))]

    def train(self, train_ds, val_ds, epochs):
        self.model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    def save(self, path: str):
        self.model.save_weights(f"saves/model-{int(time.time())}.weights.h5")
