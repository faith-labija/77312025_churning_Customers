def create_model(input_shape, hidden_layer_sizes=(50, 50), activation='relu', solver='adam', alpha=0.0001):
    inputs = Input(shape=(input_shape,))
    x = Dense(hidden_layer_sizes[0], activation=activation)(inputs)

    if len(hidden_layer_sizes) > 1:
        for units in hidden_layer_sizes[1:]:
            x = Dense(units, activation='relu')(x)

    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=solver, loss='binary_crossentropy', metrics=['accuracy'])
    return model
