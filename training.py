#import os
#os.environ["TF_CUDNN_USE_FRONTEND"] = "0"
#os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

#import tensorflow as tf
#gpus = tf.config.list_physical_devices('GPU')
#for gpu in gpus:
    #tf.config.experimental.set_memory_growth(gpu,True)

import tensorflow as tf
strategy = tf.distribute.MirroredStrategy() # I can have access to 2 CPUs

def plot_history(history):
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./outputs/loss_curve.png')
    plt.close()

# First, we train the model while freezing the base feature layers
with strategy.scope():
    from model import *
    base_model.trainable = False
    
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    
    checkpoint_1 = keras.callbacks.ModelCheckpoint(
        "./outputs/model_phase1.keras",
        monitor='val_loss',
        save_best_only=True,
    )
    
    epochs = 1
    history_1 = model.fit(train_ds, epochs=epochs, validation_data=validation_ds, callbacks=[checkpoint_1])
    
    # Save model after initial training (frozen base)
    model.save("./outputs/model_phase1.keras")
    print("Phase 1 model saved to ./outputs/model_phase1.keras")
    
    plot_history(history_1)
    
    # Now we unfreeze the base layers
    
    base_model.trainable = True
    model.summary()
    
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    
    
    checkpoint_2 = keras.callbacks.ModelCheckpoint(
        "./outputs/model_phase2.keras",
        monitor='val_loss',
        save_best_only=True,
    )
    
    epochs = 1
    history_2 = model.fit(train_ds, epochs=epochs, validation_data=validation_ds, callbacks=[checkpoint_2])
    
    # Save model after it was unfrozen
    model.save("./outputs/model_phase2.keras")
    print("Phase 2 model saved to ./outputs/model_phase1.keras")
    
    print("Evolution of accuracy before unfreezing :")
    plot_history(history_1)
    print("Evolution of accuracy after unfreezing :")
    plot_history(history_2)
