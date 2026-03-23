import numpy as np
import pandas as pd
import keras
from sklearn.metrics import f1_score
 
from load import *
 
# Build test dataset using the same load_image helper (dummy label 0)
NUM_TEST_IMAGES = 9634
 
test_ds = tf.data.Dataset.from_tensor_slices((list(range(NUM_TEST_IMAGES)), [0] * NUM_TEST_IMAGES))
test_ds = test_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
 
# Run predictions and build couples list : (image_number, class_index)
def build_predictions(model, ds=None):
    if ds is None:
        ds = test_ds
    couples = []
    for i, (images, _) in enumerate(ds):
        logits = model.predict_on_batch(images)
        predicted_classes = np.argmax(logits, axis=1)
        for j, class_index in enumerate(predicted_classes):
            couples.append((i * BATCH_SIZE + j, int(class_index)))
     
    print(f"Predicted {len(couples)} images.")
    return couples
 
 
def compute_f1(couples, ds):
    y_pred = [class_index for _, class_index in sorted(couples, key=lambda x: x[0])]
    y_true = []
    for _, labels in ds:
        y_true.extend(labels.numpy().tolist())
 
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: {len(y_true)} true labels vs {len(y_pred)} predictions. "
            "couples must be built using the same dataset passed as ds."
        )
 
    score = f1_score(y_true, y_pred, average="macro")
    print(f"Macro-averaged F1 score: {score:.4f}")
    return score

def save_csv(couples, name:str):
    rows = [{"ID": f"test_{str(img_number).zfill(5)}.png", "label": LABELS[class_index]} for img_number, class_index in couples]
    
    df = pd.DataFrame(rows, columns=["ID", "label"])
    output_path = "./outputs/" + name + ".csv"
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    for epoch in range(10):
        print(f"Model loaded from ./outputs/model_phase1_epoch{epoch}.keras")
        model = keras.models.load_model(f"./outputs/model_phase1_epoch{epoch}.keras")
        val_couples = build_predictions(model, validation_ds)
        print(compute_f1(val_couples, validation_ds))
        print("-------------------")
    for epoch in range(5):
        print(f"Model loaded from ./outputs/model_phase2_epoch{epoch}.keras")
        model = keras.models.load_model(f"./outputs/model_phase2_epoch{epoch}.keras")
        val_couples = build_predictions(model, validation_ds)
        print(compute_f1(val_couples, validation_ds))
        print("-------------------")
    #model = keras.models.load_model("./outputs/model_phase2_epoch3.keras")
    #couples ) build_predictions(model)
    #save_csv(couples, predictions)

