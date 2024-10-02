import gc

import tensorflow
from keras import Sequential
from keras.layers import Conv2D, ReLU, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from keras.saving.save import load_model
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras import layers, callbacks
from tensorflow.keras import models, optimizers
import tensorflow.keras.metrics as metrics
import os

from GANterfactual.preprocessor import preprocess_inbreast_for_classifier
from preprocessor import preprocess_inbreast_for_classifier, tf_flip_square, preprocess_vindr_for_classifier

def build_classifier():
    # Load ConvNeXtTiny model without the top layers (so we can adapt the input shape)
    base_model = ResNet50V2(include_top=False, weights=None, input_shape=(256, 256, 3))

    # Create a custom input for grayscale (1 channel) images
    input_layer = layers.Input(shape=(512, 512, 1))
    x = layers.Conv2D(3, kernel_size=3, strides=(2, 2), padding='same')(input_layer)


    # Pass the converted input through the ConvNeXt base model
    x = base_model(x)

    # Add your own layers on top of the base model here, depending on your specific problem

    # Add global pooling and the final classification head or other output layers you need
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dropout(.3)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    output_layer = layers.Dense(1, activation='sigmoid')(x)

    # Create the final model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    opt = optimizers.Adam(0.0001)
    model.compile(loss='binary_crossentropy',
                  metrics=['accuracy', metrics.Precision(), metrics.Recall()],
                  optimizer=opt)

    return model


if __name__ == "__main__":
    model = build_classifier()
    model.summary()


    dataset= preprocess_inbreast_for_classifier('train')
    dataset = dataset.batch(64)
    for mul in range(20):
        model.fit(dataset,
                  epochs=10,
                  callbacks=[callbacks.TensorBoard(log_dir="log")])
        model.save(os.path.join('..', 'models', 'classifier_inbreast', f'model_{(mul + 1) * 10}.h5'), include_optimizer=True)
    del dataset
    gc.collect() # free up some memory


    test_dataset = preprocess_inbreast_for_classifier('test')
    #model = load_model(os.path.join('..', 'models', 'classifier_vindr', 'model_100.h5'))
    model.evaluate(test_dataset.batch(32))