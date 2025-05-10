import time
import tensorflow as tf
from model import Model


def main():
    BATCH_SIZE = 32
    IMG_HEIGHT = 28
    IMG_WIDTH = 28
    SEED = int(time.time())
    VALIDATION_SPLIT = 0.1
    EPOCHS = 20
    DS_DIR = "dataset/used/"

    # load dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DS_DIR,
        seed=SEED,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DS_DIR,
        seed=SEED,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print(f"Class names = {class_names}")

    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    model = Model(load=False)
    model.summary()
    return
    model.train(train_ds, val_ds, EPOCHS)
    model.save(f"saves/model-{int(time.time())}.weights.h5")


if __name__ == "__main__":
    main()
