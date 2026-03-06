import tensorflow as tf
import pandas as pd
import random

# List of labels
LABELS = ["SNE", "LY", "MO", "EO", "BA", "VLY", "BNE", "MMY", "MY", "PMY", "BL", "PC", "PLY"]
label_to_index = {label: idx for idx, label in enumerate(LABELS)}

# Load metadata CSV
metadata = pd.read_csv("./IMA205-challenge/train_metadata.csv")
# metadata columns: ID, label

# Build list of (image_number, label_number) couples
couples = []
for _, row in metadata.iterrows():
    img_name = row["ID"] # for instance it can be : "train_00000.png"
    img_number = int(img_name.split("_")[1].split(".")[0]) # for instance here it would be 0 (the integer)
    label_number = label_to_index[row["label"]]
    couples.append((img_number, label_number))

# Shuffling : we don't want bias between the split for the training and the split for the validation
random.shuffle(couples)

# Split 80/20 (rule of thumb)
split_idx = int(len(couples) * 0.8)
train_couples = couples[:split_idx]
validation_couples   = couples[split_idx:]

# Helper: load image from number
def load_image(img_number, label):
    num_str = tf.strings.as_string(img_number, width=5, fill='0')
    path = tf.strings.join(["./IMA205-challenge/train/train_", num_str, ".png"])
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    return image, label

def make_dataset(couples): # this is just for adjusting to the required format for keras
    numbers = [c[0] for c in couples]
    labels  = [c[1] for c in couples]
    ds = tf.data.Dataset.from_tensor_slices((numbers, labels))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds

train_ds      = make_dataset(train_couples)
validation_ds = make_dataset(validation_couples)

print("Number of training samples:   %d" % tf.data.experimental.cardinality(train_ds))
print("Number of validation samples: %d" % tf.data.experimental.cardinality(validation_ds))
