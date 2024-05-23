# Define a simple classifier
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(combined_features.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(clarify_train_df['answer'].unique()), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
