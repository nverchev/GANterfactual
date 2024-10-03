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


def plot_counterfactual(array, center_lesion_x, center_lesion_y, square, fake, id_str, folder):
    width, height = 512, 512  # Dimensions of the square to draw around the lesion

    # Create the plot
    fig, ax = plt.subplots(1, 3, dpi=600)

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
    # Display the image
    ax[2].imshow(fake, cmap='gray', vmin=0, vmax=1)

    # Remove axes for cleaner display
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')

    os.makedirs(os.path.join("images", folder), exist_ok=True)
    fig.savefig(f'{os.path.join("images", folder, "counterfactual_" + id_str)}.png')
    plt.close(fig)


def plot_original(array, center_lesion_x, center_lesion_y, square, id_str, folder):
    width, height = 512, 512  # Dimensions of the square to draw around the lesion

    # Create the plot
    fig, ax = plt.subplots(1, 2, dpi=600, width_ratios=(.66, .33))

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
    fig.savefig(f'{os.path.join("images", folder, "orig_" + id_str)}.png')
    plt.close(fig)

def batch_process_vindr(batch_size=64):
    gan = CycleGAN()
    classifier_path = os.path.join('..', 'models', 'classifier_vindr', 'model_200.h5')
    ganterfactual_folder = os.path.join('..', 'models', 'GANterfactual_vindr')
    gan.load_existing(ganterfactual_folder, classifier_path=classifier_path)

    findings = vindr_findings('val')
    probs_list = []
    batch_data = []  # Store batch data
    findings = findings[['path', 'label', "Manufacturer's Model Name", 'center_lesion_x', 'center_lesion_y']]
    for i, sample in enumerate(findings.values):

        malignant = sample[1]
        array, square, center_lesion_x, center_lesion_y = process_data_vindr(sample)
        batch_data.append((array, square, center_lesion_x, center_lesion_y, malignant, i))

        # If we've accumulated a batch of 64 samples, process them
        if len(batch_data) == batch_size:
            squares = np.stack([data[1] for data in batch_data])
            probs, fakes, fake_probs = gan.predict(squares)  # Predict batch

            for j, (array, square, center_lesion_x, center_lesion_y, malignant, idx) in enumerate(batch_data):
                plot_original(array, center_lesion_x, center_lesion_y, square, id_str=f'{idx}_{j}', folder='vindr')
                plot_counterfactual(array, center_lesion_x, center_lesion_y, square, fakes[j], id_str=f'{idx}_{j}', folder='vindr')
                probs_list.append((malignant, probs[j][0], fake_probs[j][0]))

            batch_data.clear()  # Clear batch after processing

    # Process the remaining samples (if the total isn't a multiple of batch_size)
    if batch_data:
        squares = np.stack([data[1] for data in batch_data])
        probs, fakes, fake_probs = gan.predict(squares)  # Predict the remaining batch

        for j, (array, square, center_lesion_x, center_lesion_y, malignant, idx) in enumerate(batch_data):
            plot_original(array, center_lesion_x, center_lesion_y, square, id_str=f'{idx}_{j}', folder='vindr')
            plot_counterfactual(array, center_lesion_x, center_lesion_y, square, fakes[j], id_str=f'{idx}_{j}', folder='vindr')
            probs_list.append((malignant, probs[j][0], fake_probs[j][0]))

    # Save the probabilities to CSV
    prob_df = pd.DataFrame(probs_list, columns=['malignant', 'prob', 'fake_class_prob'])
    prob_df.to_csv('probs_vindr.csv')

def batch_process_inbreast(malignant_target=True, batch_size=64):
    gan = CycleGAN()
    classifier_path = os.path.join('..', 'models', 'classifier_inbreast', 'model_200.h5')
    ganterfactual_folder = os.path.join('..', 'models', 'GANterfactual_inbreast')
    gan.load_existing(ganterfactual_folder, classifier_path=classifier_path)

    probs_list = []
    batch_data = []  # Store batch data

    findings = inbreast_findings(split='test')

    for i, (image_file, malignant, xml_file) in enumerate(findings.values[:76]):
        image = read_image(image_file)
        for roi, center_lesion_x, center_lesion_y in extract_squares_inbreast(image, xml_file):
            batch_data.append((image, roi, center_lesion_x, center_lesion_y, malignant.item(), i))

            # If we've accumulated a batch of 64 samples, process them
            if len(batch_data) == batch_size:
                rois = np.stack([data[1] for data in batch_data])
                probs, fakes, fake_probs = gan.predict(rois, malignant_target=malignant_target)  # Predict batch

                for j, (image, roi, center_lesion_x, center_lesion_y, malignant_bool, idx) in enumerate(batch_data):
                    plot_original(image, center_lesion_x, center_lesion_y, roi, id_str=f'{idx}_{j}', folder='inbreast')
                    plot_counterfactual(image, center_lesion_x, center_lesion_y, roi, fakes[j], id_str=f'{idx}_{j}', folder='inbreast')
                    probs_list.append((idx, j, malignant_bool, probs[j][0], fake_probs[j][0]))

                batch_data.clear()  # Clear batch after processing

    # Process the remaining samples (if the total isn't a multiple of batch_size)
    # if batch_data:
    #     rois = np.stack([data[1] for data in batch_data])
    #     probs, fakes, fake_probs = gan.predict(rois, malignant_target=malignant_target)  # Predict the remaining batch
    #
    #     for j, (image, roi, center_lesion_x, center_lesion_y, malignant, idx) in enumerate(batch_data):
    #         plot_original(image, center_lesion_x, center_lesion_y, roi, id_str=f'{idx}_{j}', folder='inbreast')
    #         plot_counterfactual(image, center_lesion_x, center_lesion_y, roi, fakes[j], id_str=f'{idx}_{j}', folder='inbreast')
    #         probs_list.append((idx, j, malignant, probs[j][0], fake_probs[j][0]))

    # Save the probabilities to CSV
    prob_df = pd.DataFrame(probs_list, columns=['image_id','batch_index','malignant', 'prob', 'fake_class_prob'])
    prob_df.to_csv('probs_inbreast.csv')


if __name__ == '__main__':
    batch_process_inbreast(malignant_target=True)



