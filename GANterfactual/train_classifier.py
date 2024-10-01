import gc

import tensorflow
from keras import Sequential
from keras.layers import Conv2D, ReLU, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from keras.saving.save import load_model
from tensorflow.keras.applications import ConvNeXtTiny
from tensorflow.keras import layers, callbacks
from tensorflow.keras import models, optimizers
import tensorflow.keras.metrics as metrics
import os

from GANterfactual.preprocessor import preprocess_inbreast_for_classifier
from preprocessor import preprocess_inbreast_for_classifier, tf_flip_square, preprocess_vindr_for_classifier
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

# def build_classifier():
#     # Load ConvNeXtTiny model without the top layers (so we can adapt the input shape)
#     base_model = ConvNeXtTiny(include_top=False, weights=None, input_shape=(256, 256, 3))
#
#     # Create a custom input for grayscale (1 channel) images
#     input_layer = layers.Input(shape=(512, 512, 1))
#
#     # Convert the grayscale input to a 3-channel input (by repeating across the channels)
#     # You can also just leave it as is if you prefer, but ConvNeXt expects 3 channels.
#     x = layers.Conv2D(3, (3, 3), padding='same', strides=(2, 2))(input_layer)
#
#     # Pass the converted input through the ConvNeXt base model
#     x = base_model(x)
#
#     # Add global pooling and the final classification head or other output layers you need
#     x = layers.GlobalAveragePooling2D()(x)
#     x = layers.Dense(128, activation='relu')(x)
#     x = layers.Dense(32, activation='relu')(x)  # Example: 10-class classification
#     output_layer = layers.Dense(1, activation='linear')(x)  # Example: 10-class classification
#
#     # Create the final model
#     model = models.Model(inputs=input_layer, outputs=output_layer)
#
#     opt = optimizers.Adam(0.001)
#     model.compile(loss=tensorflow.keras.losses.BinaryCrossentropy(from_logits=True),
#                   metrics=['accuracy', metrics.Precision(class_id=0), metrics.Recall(class_id=0)],
#                   optimizer=opt)
#
#     return model

def build_classifier():
    img_shape= (512, 512, 1)
    # Load ConvNeXtTiny model without the top layers (so we can adapt the input shape)
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


    # Passing it to a dense layer
    model.add(Flatten())

    # 3rd Dense Layer
    model.add(Dense(1000))
    model.add(ReLU())
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Output Layer
    model.add(Dense(1))
    opt = optimizers.Adam(0.001)
    model.compile(loss=tensorflow.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy', metrics.Precision(class_id=0), metrics.Recall(class_id=0)],
                  optimizer=opt)

    return model


if __name__ == "__main__":
    model = build_classifier()
    model.summary()
    #

    dataset= preprocess_vindr_for_classifier('train')
    dataset = dataset.batch(32)
    for mul in range(10):
        model.fit(dataset,
                  epochs=10,
                  callbacks=[callbacks.TensorBoard(log_dir="log")])
        model.save(os.path.join('..', 'models', 'classifier_vindr', f'model_{(mul + 1) * 10}.h5'), include_optimizer=True)
    del dataset
    gc.collect() # free up some memory


    test_dataset = preprocess_vindr_for_classifier('val')
    model.load_weights(os.path.join('..', 'models', 'classifier_final', 'model_50.h5'))
    model.evaluate(test_dataset.batch(32))