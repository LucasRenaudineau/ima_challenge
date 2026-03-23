import tensorflow as tf
import pandas as pd
import random
import matplotlib.pyplot as plt
from collections import defaultdict


# List of labels
LABELS = ["SNE", "LY", "MO", "EO", "BA", "VLY", "BNE", "MMY", "MY", "PMY", "BL", "PC", "PLY"]
label_to_index = {label: idx for idx, label in enumerate(LABELS)}

def oversample(couples):
    # Group by class
    by_class = defaultdict(list)
    for img_number, label_index in couples:
        by_class[label_index].append((img_number, label_index))
 
    max_count = max(len(v) for v in by_class.values())
 
    print("Oversampling training set:")
    oversampled = []
    for label_index, samples in sorted(by_class.items()):
        # Repeat the list as many times as needed, then trim to max_count
        repeated = (samples * (max_count // len(samples) + 1))[:max_count]
        print(f"  {LABELS[label_index]:>4s}: {len(samples):>5d} → {len(repeated)}")
        oversampled.extend(repeated)
 
    random.shuffle(oversampled)
    return oversampled
 
# Helper: for the make_dataset function
def load_image(img_number, label_index):
    num_str = tf.strings.as_string(img_number, width=5, fill='0')
    path = tf.strings.join(["./IMA205-challenge/train/train_", num_str, ".png"])
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [368,368])
    return image, label_index

BATCH_SIZE = 16

def make_dataset(couples): # this is just for adjusting to the required format for keras and make batches
    numbers = [c[0] for c in couples]
    labels  = [c[1] for c in couples]
    ds = tf.data.Dataset.from_tensor_slices((numbers, labels))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def print_couple(image, label):
    plt.imshow(image)
    plt.title(int(label))
    plt.axis("off")

def print_first_in_dataset(dataset, n, filename="output.png"):
    plt.figure(figsize=(10, 10))
    for i, (image, label_index) in enumerate(dataset.take(n)):
        ax = plt.subplot(2, 2, i + 1)
        plt.imshow(image.numpy())
        plt.title(LABELS[int(label_index)])
        plt.axis("off")
    plt.savefig(filename)
    print(f"Saved to {filename}")
    plt.close()

def load_data():

    # Load metadata CSV
    metadata = pd.read_csv("./IMA205-challenge/train_metadata.csv")
    
    # Builds a list of couples in the form of : (image_number, label_index)
    couples = []
    for _, row in metadata.iterrows():
        img_name = row["ID"] # for instance it can be : "train_00000.png"
        img_number = int(img_name.split("_")[1].split(".")[0]) # for instance here it would be 0 (the integer)
        label_index = label_to_index[row["label"]]
        couples.append((img_number, label_index))


    # This is to verify it did not take the title row with "ID" and "LABEL"
    print(couples[0])
    print(couples[1])
    print(couples[-1])
    
    # Shuffling : we don't want bias between the split for the training and the split for the validation
    random.shuffle(couples)
    
    # Split 80/20 (rule of thumb)
    split_idx = int(len(couples) * 0.8)
    train_couples = couples[:split_idx]
    validation_couples   = couples[split_idx:]
    train_couples_oversampled = oversample(train_couples)
    # train_ds = make_dataset(train_couples_oversampled) # commented for the moment
    train_ds = make_dataset(train_couples)
    validation_ds = make_dataset(validation_couples)
    return train_ds, validation_ds

if __name__ == "__main__":
    print("hello world")

    train_ds, validation_ds = load_data()
    
    print_first_in_dataset(train_ds, 4, "./outputs/train_preview.png")
    print_first_in_dataset(validation_ds, 4, "./outputs/validation_preview.png")
 
    print("Should adds up to 28901 :")
    print("Number of training samples:   %d" % tf.data.experimental.cardinality(train_ds))
    print("Number of validation samples: %d" % tf.data.experimental.cardinality(validation_ds))
