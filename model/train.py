from tensorflow.keras.callbacks import ModelCheckpoint

def train_model(model, train_data, validation_data, epochs=20, batch_size=20):
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # Checkpoint callback to save the best model during training
    checkpointer = ModelCheckpoint(filepath='./weights.best.VGG16.keras', 
                                   monitor='val_accuracy',  # Monitoring validation accuracy
                                   verbose=1, 
                                   save_best_only=True,  # Save only the best model
                                   mode='max')  # We want to maximize validation accuracy

    # Train the model
    model.fit(train_data,
              validation_data=validation_data,
              epochs=epochs, batch_size=batch_size,
              callbacks=[checkpointer], verbose=1)