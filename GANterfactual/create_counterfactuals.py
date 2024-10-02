import os
import numpy as np
import pandas as pd
import pydicom

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from GANterfactual.cyclegan_breast import CycleGAN
from GANterfactual.preprocessor import vindr_findings, window_image, extract_square_from_center, random_square_crop, \
    read_image, extract_squares_inbreast
from GANterfactual.preprocessor import inbreast_findings





def process_data_vindr(sample):
    image_path, label, scanner, center_lesion_x, center_lesion_y = sample

    # Use pydicom to read the DICOM file from byte data
    dicom = pydicom.dcmread(image_path)

    array = dicom.pixel_array
    if scanner == 'Planmed Nuance':
        array = (np.max(array) - array) / 4

    array = window_image(array, window_size=500)
    if not np.isnan(center_lesion_x) and not np.isnan(center_lesion_y):
        center_lesion_x = int(center_lesion_x)
        center_lesion_y = int(center_lesion_y)
        center = (center_lesion_x, center_lesion_y)
        square = extract_square_from_center(array, center)

    else:
        square, top_x, top_y = random_square_crop(array)
        center_lesion_x = top_x + 256
        center_lesion_y = top_y + 256
    return array, square, center_lesion_x, center_lesion_y


def plot_figure(array, center_lesion_x, center_lesion_y, square, name, folder):
    width, height = 512, 512  # Dimensions of the square to draw around the lesion

    # Create the plot
    fig, ax = plt.subplots(1, 2, figsize=(array.shape[0] / 100, array.shape[1] / 100))

    ax[0].imshow(array, cmap='gray', vmin=0, vmax=1)
    #
    x = center_lesion_x - width // 2 # x-coordinate of the top-left corner
    y = center_lesion_y - width // 2
    # Create a Rectangle patch
    rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')

    # Add the rectangle to the plot
    ax[0].add_patch(rect)

    # Display the image
    ax[1].imshow(square, cmap='gray', vmin=0, vmax=1)

    # Remove axes for cleaner display
    ax[0].axis('off')
    ax[1].axis('off')
    os.makedirs(os.path.join("images", folder), exist_ok=True)
    fig.savefig(f'{os.path.join("images", folder, name)}.png')
    plt.close(fig)

def vindr_counterfactuals():
    gan = CycleGAN()
    classifier_path = os.path.join('..', 'models', 'classifier_vindr', 'model_100.h5')
    ganterfactual_folder = os.path.join('..', 'models', 'GANterfactual', 'ep_49')
    gan.load_existing(ganterfactual_folder, classifier_path=classifier_path)

    findings = vindr_findings('val')
    findings = findings[['path', 'label', "Manufacturer's Model Name", 'center_lesion_x', 'center_lesion_y']]
    probs_list = []
    for i, sample in enumerate(findings.values):
        malignant = sample[1]
        array, square, center_lesion_x, center_lesion_y = process_data_vindr(sample)
        plot_figure(array, center_lesion_x, center_lesion_y, square, name='orig_' + str(i), folder='vindr')
        prob, fake, fake_prob= gan.predict(square, positive_sample=malignant)
        plot_figure(array, center_lesion_x, center_lesion_y, fake[0], name='counterfactual_' + str(i), folder='vindr')
        probs_list.append((malignant, prob[0], fake_prob[0]))
    prob_df = pd.DataFrame(probs_list, columns=['malignant', 'prob', 'fake_class_prob'])
    prob_df.to_csv('probs.csv')

def inbreast_counterfactuals():
    gan = CycleGAN()
    classifier_path = os.path.join('..', 'models', 'classifier_vindr', 'model_100.h5')
    ganterfactual_folder = os.path.join('..', 'models', 'GANterfactual', 'ep_49')
    gan.load_existing(ganterfactual_folder, classifier_path=classifier_path)

    findings = inbreast_findings(split='test')
    probs_list = []
    for i, (image_file, malignant, xml_file) in enumerate(findings.values):
        image = read_image(image_file)
        for roi, center_lesion_x, center_lesion_y  in extract_squares_inbreast(image, xml_file):
            plot_figure(image, center_lesion_x, center_lesion_y, roi, name='orig_' + str(i), folder='inbreast')
            prob, fake, fake_prob = gan.predict(roi, positive_sample=malignant)
            plot_figure(image, center_lesion_x, center_lesion_y, fake[0], name='counterfactual_' + str(i), folder='inbreast')
            probs_list.append((malignant, prob[0], fake_prob[0]))
        prob_df = pd.DataFrame(probs_list, columns=['malignant', 'prob', 'fake_class_prob'])
        prob_df.to_csv('probs.csv')


if __name__ == '__main__':
    inbreast_counterfactuals()


