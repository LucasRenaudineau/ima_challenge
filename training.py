#import os
#os.environ["TF_CUDNN_USE_FRONTEND"] = "0"
#os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

#import tensorflow as tf
#gpus = tf.config.list_physical_devices('GPU')
#for gpu in gpus:
    #tf.config.experimental.set_memory_growth(gpu,True)

from model import *

# First, we train the model while freezing the base feature layers
base_model.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

epochs = 1
model.fit(train_ds, epochs=epochs, validation_data=validation_ds)

# Save model after initial training (frozen base)
model.save("./outputs/model_phase1.keras")
print("Phase 1 model saved to ./outputs/model_phase1.keras")

# Now we unfreeze the base layers

base_model.trainable = True
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

epochs = 1
model.fit(train_ds, epochs=epochs, validation_data=validation_ds)

x_train = list(map(lambda x: x[0], train_ds))
y_train = list(map(lambda x: x[1], train_ds))
x_validation = list(map(lambda x: x[0], validation_ds))
y_validation = list(map(lambda x: x[1], validation_ds))

# Save model after it was unfrozen
model.save("./outputs/model_phase2.keras")
print("Phase 2 model saved to ./outputs/model_phase1.keras")
