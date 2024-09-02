
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, ReLU, BatchNormalization, Dropout

# The trained classifier is loaded.
# Rewrite this function if you want to use another model architecture than our modified AlexNET.
# A model, which provides a 'predict' function, has to be returned.
def load_classifier(path, img_shape):
    original = load_model(path)
    classifier = build_classifier(img_shape)

    counter = 0
    for layer in original.layers:
        assert (counter < len(classifier.layers))
        classifier.layers[counter].set_weights(layer.get_weights())
        counter += 1

    classifier.summary()

    return classifier

def build_classifier(img_shape):
    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=96,
                     kernel_size=(11, 11),
                     strides=(4, 4),
                     padding='valid',
                     input_shape=img_shape))
    model.add(ReLU())
    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    # Batch Normalisation before passing it to the next layer
    model.add(BatchNormalization())

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256,
                     kernel_size=(11, 11),
                     strides=(1, 1),
                     padding='valid'))
    model.add(ReLU())
    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='valid'))
    model.add(ReLU())
    # Batch Normalisation
    model.add(BatchNormalization())

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='valid'))
    model.add(ReLU())
    # Batch Normalisation
    model.add(BatchNormalization())

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='valid'))
    model.add(ReLU())
    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Passing it to a dense layer
    model.add(Flatten())
    # 1st Dense Layer
    model.add(Dense(4096))
    model.add(ReLU())
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 2nd Dense Layer
    model.add(Dense(4096))
    model.add(ReLU())
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 3rd Dense Layer
    model.add(Dense(1000))
    model.add(ReLU())
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Output Layer
    model.add(Dense(2))
    model.add(ReLU())

    return model