#import os
#os.environ["TF_CUDNN_USE_FRONTEND"] = "0"
#os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from load import *
import tensorflow as tf
from model import *
from evaluates import *
import keras

strategy = tf.distribute.MirroredStrategy() # I can have access to 2 GPUs

def plot_history(history):
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./outputs/loss_curve.png')
    plt.close()

class MacroF1Callback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        couples = build_predictions(self.model, validation_ds)
        score = compute_f1(couples, validation_ds)
        logs["val_macro_f1"] = score
        print(f"\nEpoch {epoch} — val macro F1: {score:.4f}")

# First, we train the model while freezing the base feature layers
#with strategy.scope():

def train_one_epoch(model, epoch,frozen:bool):
    # Class weights to handle imbalance
    # counts = [13015, 8101, 2746, 2012, 861, 441, 415, 391, 366, 360, 114, 68, 11]
    # total = sum(counts)
    num_classes = len(LABELS)
    # class_weights = {i: total / (num_classes * count) for i, count in enumerate(counts)}

    efficientnet_layer = model.get_layer("efficientnetb2")
    efficientnet_layer.trainable = not frozen


    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate = 1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ]
    )
    model.fit(train_ds, epochs=1, validation_data=validation_ds, callbacks=[MacroF1Callback()])

    # Save model after initial training (frozen base)
    print(f"{epoch}-th epoch is finished.")
    if frozen:
        model.save(f"./outputs/model_phase1_epoch{epoch}.keras")
        print(f"Phase 1 model saved to ./outputs/model_phase1_epoch{epoch}.keras")
    else:
        model.save(f"./outputs/model_phase2_epoch{epoch}.keras")
        print(f"Phase 2 model saved to ./outputs/model_phase2_epoch{epoch}.keras")

if __name__ == "__main__":
    with strategy.scope():
        # Loading data INSIDE STRATEGY.SCOPE
        train_ds, validation_ds = load_data()
        # Phase 1 of training (with frozen base)
        save_base_model()
        model = keras.models.load_model("./outputs/model_not_trained.keras")

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate = 1e-4),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            ]
        )
        for epoch in range(10):
            train_one_epoch(model, epoch, True)
            val_couples = build_predictions(model,validation_ds)
            score_f1 = compute_f1(val_couples, validation_ds)
            print(f"The F1 score of the epoch {epoch} is {score_f1}.")
        # Phase 2 of training (with unfrozen base)
        model = keras.models.load_model("./outputs/model_phase1_epoch6.keras") # Take the best model on the validation test !!
        for epoch in range(5):
            train_one_epoch(model,epoch, False)
            val_couples = build_predictions(model,validation_ds)
            score_f1 = compute_f1(val_couples, validation_ds)
            print(f"The F1 score of the epoch {epoch} is {score_f1}.")
