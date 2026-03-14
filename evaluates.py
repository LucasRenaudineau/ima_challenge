import numpy as np
import pandas as pd
import keras
 
from load import *
 
# Load the saved model
model = keras.models.load_model("./outputs/model_final.keras")
print("Model loaded from ./outputs/model_final.keras")
 
# Build test dataset using the same load_image helper (dummy label 0)
NUM_TEST_IMAGES = 9634
 
test_ds = tf.data.Dataset.from_tensor_slices((list(range(NUM_TEST_IMAGES)), [0] * NUM_TEST_IMAGES))
test_ds = test_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(32).prefetch(tf.data.AUTOTUNE)
 
# Run predictions and build couples list : (image_number, class_index)
couples = []
for i, (images, _) in enumerate(test_ds):
    logits = model.predict_on_batch(images)
    predicted_classes = np.argmax(logits, axis=1)
    for j, class_index in enumerate(predicted_classes):
        couples.append((i * 32 + j, int(class_index)))
 
print(f"Predicted {len(couples)} images.")
 
# Build and save CSV
rows = [{"": img_number, "ID": f"test_{str(img_number).zfill(5)}.png", "label": LABELS[class_index]}
        for img_number, class_index in couples]
 
df = pd.DataFrame(rows, columns=["", "ID", "label"])
output_path = "./outputs/predictions.csv"
df.to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")
