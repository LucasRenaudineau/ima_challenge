import numpy as np
import pandas as pd
import keras
from sklearn.metrics import f1_score
 
from load import *
 
# Load the saved model
model = keras.models.load_model("./outputs/model_phase1.keras")
print("Model loaded from ./outputs/model_phase1.keras")
 
# Build test dataset using the same load_image helper (dummy label 0)
NUM_TEST_IMAGES = 9634
 
test_ds = tf.data.Dataset.from_tensor_slices((list(range(NUM_TEST_IMAGES)), [0] * NUM_TEST_IMAGES))
test_ds = test_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(32).prefetch(tf.data.AUTOTUNE)
 
# Run predictions and build couples list : (image_number, class_index)
def build_predictions(model):
    couples = []
    for i, (images, _) in enumerate(test_ds):
        logits = model.predict_on_batch(images)
        predicted_classes = np.argmax(logits, axis=1)
        for j, class_index in enumerate(predicted_classes):
            couples.append((i * 32 + j, int(class_index)))
     
    print(f"Predicted {len(couples)} images.")
    return couples

def compute_f1(model):
    y_true = []
    y_pred = []
    for images, labels in validation_ds:
        logits = model.predict_on_batch(images)
        predicted_classes = np.argmax(logits, axis=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(predicted_classes.tolist())
    
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
        couples = build_predictions(model)
        print(compute_f1(model))
        print("-------------------")
    for epoch in range(5):
        print(f"Model loaded from ./outputs/model_phase2_epoch{epoch}.keras")
        model = keras.models.load_model(f"./outputs/model_phase2_epoch{epoch}.keras")
        couples = build_predictions(model)
        print(compute_f1(model))
        print("-------------------")
    #model = keras.models.load_model("./outputs/model_phase2_epoch3.keras")
    #couples ) build_predictions(model)
    #save_csv(couples, predictions)

