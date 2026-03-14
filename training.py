from model import *

# First, we train the model while freezing the base feature layers
base_model.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

epochs = 10
model.fit(train_ds, epochs=epochs, validation_data=validation_ds)

# Save model after initial training (frozen base)
model.save("./outputs/model_phase1.keras")
print("Phase 1 model saved to ./outputs/model_phase1.keras")

# Now we unfreeze the base layers

base_model.trainable = True
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

epochs = 5
model.fit(train_ds, epochs=epochs, validation_data=validation_ds)

x_train = list(map(lambda x: x[0], train_ds))
y_train = list(map(lambda x: x[1], train_ds))
x_validation = list(map(lambda x: x[0], validation_ds))
y_validation = list(map(lambda x: x[1], validation_ds))

# Save model after it was unfrozen
model.save("./outputs/model_phase2.keras")
print("Phase 2 model saved to ./outputs/model_phase1.keras")
