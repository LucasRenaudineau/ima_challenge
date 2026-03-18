# Libraries
import keras
from tensorflow.keras import layers
import numpy as np

# Files
from load import *

data_augmentation = keras.Sequential(
    [layers.RandomFlip("horizontal"), layers.RandomFlip("vertical"), layers.RandomRotation(0.5),]
)
# For the choice of data augmentation transformations, I searched there : https://www.tensorflow.org/tutorials/images/data_augmentation

def print_first_in_dataset_augmented(dataset, n=9):
    for i, (image, label_index) in enumerate(dataset.take(n)):
        plt.figure(figsize=(10, 10))
        for j in range(n):
            ax = plt.subplot(3, 3, j + 1)
            augmented_image = data_augmentation(
                tf.expand_dims(image, 0), training=True
            )
            plt.imshow(augmented_image[0].numpy().astype("int32"))
            plt.title(LABELS[int(label_index)])
            plt.axis("off")
        filename = f"./outputs/augmentation_{i}.png"
        plt.savefig(filename)
        print(f"Saved to {filename}")
        plt.close()

if __name__ == "__main__":
    print_first_in_dataset_augmented(train_ds,9)
