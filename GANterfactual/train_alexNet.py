import gc

from sympy.printing.tensorflow import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ReLU, Dropout, Flatten, Conv2D, MaxPooling2D, Softmax
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras import callbacks, optimizers
import tensorflow.keras.metrics as metrics
import os

from preprocessor import preprocess_inbreast_for_classifier, tf_flip_square


def get_adapted_alexNet():
    dimension = 512
    model = Sequential()
    # 1st Convolutional Layer
    model.add(Conv2D(filters=96,
                     kernel_size=(11, 11),
                     strides=(4, 4),
                     padding='valid',
                     kernel_regularizer=l2(0.001),
                     bias_regularizer=l2(0.001),
                     input_shape=(dimension, dimension, 1)))
    model.add(ReLU())
    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    # Batch Normalization before passing it to the next layer
    model.add(BatchNormalization())

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256,
                     kernel_size=(11, 11),
                     strides=(1, 1),
                     padding='valid',
                     kernel_regularizer=l2(0.001),
                     bias_regularizer=l2(0.001)))
    model.add(ReLU())
    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    # Batch Normalization
    model.add(BatchNormalization())

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='valid'))
    model.add(ReLU())
    # Batch Normalization
    model.add(BatchNormalization())

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='valid',
                     kernel_regularizer=l2(0.001),
                     bias_regularizer=l2(0.001)))
    model.add(ReLU())
    # Batch Normalization
    model.add(BatchNormalization())

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='valid'))
    model.add(ReLU())
    # Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    # Batch Normalization
    model.add(BatchNormalization())

    # Passing it to a dense layer
    model.add(Flatten())
    # 1st Dense Layer
    model.add(Dense(4096, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))
    model.add(ReLU())
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))
    # Batch Normalization
    model.add(BatchNormalization())

    # 2nd Dense Layer
    model.add(Dense(4096, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))
    model.add(ReLU())
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalization
    model.add(BatchNormalization())

    # 3rd Dense Layer
    model.add(Dense(1000, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))
    model.add(ReLU())
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalization
    model.add(BatchNormalization())

    # Output Layer
    model.add(Dense(2))
    model.add(Softmax())


    opt = optimizers.SGD(0.001, 0.9)
    model.compile(loss='binary_crossentropy',
                  metrics=['accuracy', metrics.Precision(class_id=1), metrics.Recall(class_id=1)],
                  optimizer=opt)

    return model




if __name__ == "__main__":
    model = get_adapted_alexNet()
    model.summary()
    test_dataset = preprocess_inbreast_for_classifier('test')

    dataset= preprocess_inbreast_for_classifier('trainval')
    dataset = dataset.map(tf_flip_square).batch(32)
    model.fit(dataset,
              epochs=200,
              callbacks=[callbacks.TensorBoard(log_dir="log")])


    model.save(os.path.join('..', 'models', 'classifier_final', 'model.h5'), include_optimizer=False)
    del dataset
    gc.collect() # free up some memory

    model.load_weights(os.path.join('..', 'models', 'classifier_final', 'model.h5'))
    model.evaluate(test_dataset.batch(32))