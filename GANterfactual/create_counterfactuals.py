import os

import numpy as np

from GANterfactual.cyclegan_breast import CycleGAN
from GANterfactual.preprocessor import preprocess_vindr_for_ganterfactual

def probs(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x) + 0.000001)
    return e_x[:, 1] / e_x.sum(axis=1)
import numpy as np
import pydicom
import warnings
import pathlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_figure(array, center_lesion_x, center_lesion_y, square, name):

    # Create the plot
    fig, ax = plt.subplots(1, 2, figsize=(array.shape[0] / 100, array.shape[1] / 100))

    ax[0].imshow(array, cmap='gray', vmin=0, vmax=1)
    #
    x = center_lesion_x - width // 2 - 100  # x-coordinate of the top-left corner
    y = center_lesion_y - width // 2 + 100
    # Create a Rectangle patch
    rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')

    # Add the rectangle to the plot
    ax[0].add_patch(rect)

    # Display the image
    ax[1].imshow(square, cmap='gray', vmin=0, vmax=1)

    # Remove axes for cleaner display
    ax[0].axis('off')
    ax[1].axis('off')
    fig.savefig(f'images/{name}.png')

if __name__ == '__main__':
    dataset = preprocess_vindr_for_ganterfactual('val')

    width = 512  # width of the rectangle
    height = 512  # height of the rectangle


    gan = CycleGAN()
    classifier_path = os.path.join('..', 'models', 'classifier_final', 'model_20.h5')
    ganterfactual_folder = os.path.join('..', 'models', 'GANterfactual', 'ep_49')
    gan.load_existing(ganterfactual_folder, classifier_path=classifier_path, classifier_weight=0.05)

    mapper = map(np.concatenate, gan.predict(dataset, batch_size=64))
    class_neg, class_pos, fake_P, fake_N, class_fake_P, class_fake_N = list(mapper)
    p_neg = probs(class_neg)
    p_pos = probs(class_pos)
    p_fake_P = probs(class_fake_P)
    p_fake_N = probs(class_fake_N)

    for fake in fake_P:
        plot_figure(image, 400, 400, )

    print('predicted')

