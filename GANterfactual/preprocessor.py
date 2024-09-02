import typing
import random
import numpy as np
import tensorflow
import pydicom
import pandas as pd
import os
import argparse
import dotenv
import pathlib
import xml.etree.ElementTree as ET
import cv2
from tensorflow.python.framework import dtypes


def parse_args() -> argparse.Namespace:
    try:
        data_path = os.getenv('DATASET_PATH')
    except TypeError:
        raise FileNotFoundError('File .env does not exists or it does not contain the path')
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--in', default=data_path, help='input folder')
    #ap.add_argument('-o', '--out', required=True, help='output folder')
    ap.add_argument('-t', '--test',  default=20, help='proportion of images used for test')
    ap.add_argument('-v', '--validation', default=10, help='proportion of images used for validation')
    ap.add_argument('-d', '--dimension', default=512, help='new dimension for files')
    return ap.parse_args()

def shuffled_patients_paths(dicom_paths):
    random.seed(123)
    patients_paths = {}
    for image_file in dicom_paths:
        if image_file.suffix == '.m':
            continue
        patients_paths.setdefault(image_file.name.split('_')[1], []).append(image_file)

    patients_id = list(patients_paths)
    random.shuffle(patients_id)
    shuffled_dictionary = {}
    for patient_id in patients_id:
        shuffled_dictionary[patient_id] = patients_paths[patient_id]

    return shuffled_dictionary




def create_mask_from_polygon(image_shape, polygon_coords):
    """
    Create a mask from a polygon defined by a list of coordinates.

    :param image_shape: Tuple defining the shape of the mask (height, width).
    :param polygon_coords: List of (x, y) tuples representing the vertices of the polygon.
    :return: A binary mask with the polygon filled.
    """
    # Create an empty mask with the given shape
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert the polygon coordinates to the correct format for OpenCV
    polygon = np.array(polygon_coords, dtype=np.int32)

    # Fill the polygon on the mask
    cv2.fillPoly(mask, [polygon], 1)

    return mask


def parse_dict(d):
    roi_info = {}
    elements = list(d)
    for i in range(0, len(elements), 2):
        key = elements[i].text
        value_elem = elements[i + 1]

        if value_elem.tag == 'real':
            roi_info[key] = float(value_elem.text)
        elif value_elem.tag == 'integer':
            roi_info[key] = int(value_elem.text)
        elif value_elem.tag == 'string':
            roi_info[key] = value_elem.text
        elif value_elem.tag == 'array':
            roi_info[key] = [item.text for item in value_elem.findall('string')]
    return roi_info

def extract_rois(xml_path):
    # Parse the XML data
    root = ET.parse(xml_path)
    rois = []
    for image in root.findall(".//array/dict/array/dict"):
        roi_str = parse_dict(image) ['Point_px']
        roi = [[float(coo) for coo in points.strip('()').split(', ')] for points in roi_str]
        rois.append(np.array(roi))

    return rois


def extract_muscle_rois(xml_path: pathlib.Path) -> list[np.ndarray]:
    # Parse the XML data
    root = ET.parse(xml_path)
    rois = []
    for image in root.findall(".//dict"):
        rois.append(parse_dict(image))
    try:
        muscle_roi =  [[float(coo) for coo in points.strip('{}').split(', ')] for points in rois[0]['ROIPoints']]
    except KeyError:
        muscle_roi =  [[float(coo) for coo in points.strip('()').split(', ')] for points in rois[2]['Point_px']]

    return [np.array(points) for points in muscle_roi]




def read_image(dcm_path: pathlib.Path) -> np.ndarray:
    image = pydicom.dcmread(dcm_path).pixel_array / 4095
    return image

class ImageAndMuscle(typing.NamedTuple):
    image: np.ndarray
    muscle_mask: typing.Optional[np.ndarray]


def extract_muscle_mask(image: np.ndarray, muscle_path: pathlib.Path) -> typing.Optional[np.ndarray]:
    if muscle_path.exists():
        muscle_roi = extract_muscle_rois(muscle_path)
        corner = (0, 0) if muscle_roi[0][0] < image.shape[0] / 2 else (image.shape[0], 0)
        muscle_roi.append(np.array(corner))
        return create_mask_from_polygon(image.shape, muscle_roi)
    else:
        return None

def extract_squares(image: np.ndarray, xml_path: pathlib.Path, size=512) -> list[np.ndarray]:
    # Image dimensions
    height, width = image.shape[:2]

    # Half size for easy calculations
    half_size = size // 2

    if xml_path.exists():
        rois = extract_rois(xml_path)
        squares = []
        for roi in rois:
            center = np.mean(roi, axis=0).astype(np.int32)
            # Initial boundaries based on the point being at the center
            x_center, y_center = center
            left = max(0, x_center - half_size)
            right = min(width, x_center + half_size)
            top = max(0, y_center - half_size)
            bottom = min(height, y_center + half_size)
            # Adjust boundaries if the square goes out of the image bounds
            if left == 0:
                right = size
            elif right == width:
                left = width - size
            if top == 0:
                bottom = size
            elif bottom == height:
                top = height - size

            squares.append(image[top:bottom, left:right])
        return squares
    else:
        return []


def tf_flip_square(image, label=None):
    # Wrap the numpy function to be used with TensorFlow
    image = tensorflow.image.random_flip_left_right(image)
    image = tensorflow.image.random_flip_up_down(image)
    k = tensorflow.random.uniform(shape=[], minval=0, maxval=1, dtype=dtypes.int32)
    image = tensorflow.image.rot90(image, k=k)
    if label is None:
        return image
    else:
        return image, label

def random_square_crop(data_point: ImageAndMuscle,
                       square_size=512,
                       max_black_pixel_ratio=0.3) -> tuple[np.ndarray]:
    """
    Randomly crops a square from the image and checks if it's mostly black.

    Parameters:
    - image (tf.Tensor): The input image as a TensorFlow tensor.
    - square_size (int): The size of the square to crop.
    - max_black_pixel_ratio (float): The threshold ratio for black pixels.

    Returns:
    - cropped square (tf.Tensor) or None if it’s mostly black.
    """
    image = data_point.image
    muscle_mask = data_point.muscle_mask
    height, width = image.shape[:2]
    total_pixels = square_size ** 2
    while True:
        # Randomly select top-left corner for the square crop
        y = np.random.randint(0, height - square_size)
        x = np.random.randint(0, width - square_size)

        # Remove muscle:
        image_no_muscle = (1 - muscle_mask) * image if muscle_mask is not None else image

        # Crop the square
        square = image_no_muscle[y:y + square_size, x:x + square_size]

        # Calculate the ratio of non-black pixels
        non_black_pixels = np.count_nonzero(square)

        if non_black_pixels > (1 - max_black_pixel_ratio) * total_pixels:
            square = image[y:y + square_size, x:x + square_size]
            return np.expand_dims(square, axis=-1),




class ShuffledDataset:

    def __init__(self, data_list):
        self.data_list = data_list

    def __iter__(self):
        random.shuffle(self.data_list)
        for item in self.data_list:
            yield item

def preprocess_inbreast_for_pretraining(split='trainval') -> tensorflow.data.Dataset:
    dotenv.load_dotenv()

    args = vars(parse_args())
    in_path = pathlib.Path(args['in'])
    test_size = float(args['test'])
    val_size = float(args['validation'])
    dim = int(args['dimension'])

    dicom_path = in_path / 'AllDICOMs'
    muscle_path = in_path / 'PectoralMuscle' / 'Pectoral Muscle XML'

    dcm_files = list(dicom_path.iterdir())
    patient_paths = shuffled_patients_paths(dcm_files)


    len_ds = len(patient_paths)
    val_split = int(len_ds * (100 - val_size - test_size) / 100 )
    test_split = int(len_ds * (100 - test_size) / 100 )
    if split == 'train':
        indices = slice(0, val_split)
    elif split == 'trainval':
        indices = slice(0, test_split)
    elif split == 'val':
        indices = slice(val_split, test_split)
    else:
        assert split == 'test'
        indices = slice(test_split, None)

    dataset: list[ImageAndMuscle] = []
    for patient in list(patient_paths)[indices]:
        for image_file in patient_paths[patient]:
            if image_file.suffix == '.m':
                continue
            id_str = image_file.name.split('_')[0]
            muscle_file = (muscle_path / (id_str + '_muscle')).with_suffix('.xml')

            image = read_image(image_file)
            roi = None #extract_roi(xml_file)
            muscle_mask = extract_muscle_mask(image, muscle_file)
            dataset.append(ImageAndMuscle(image, muscle_mask))

    tf_dataset =  tensorflow.data.Dataset.from_generator(lambda: map(random_square_crop, ShuffledDataset(dataset)),
                                                        output_signature=(
                                                        tensorflow.TensorSpec(shape=(dim, dim, 1), dtype=np.float32),
                                                    ))
    return tf_dataset


def preprocess_inbreast_for_classifier(split='trainval') -> tensorflow.data.Dataset:
    dotenv.load_dotenv()

    args = vars(parse_args())
    in_path = pathlib.Path(args['in'])
    test_size = float(args['test'])
    val_size = float(args['validation'])
    dim = int(args['dimension'])

    xls_sheet = pd.read_excel(in_path / 'INbreast.xls', sheet_name="Sheet1", dtype=str)[:-2]
    xls_sheet['malignant'] = xls_sheet['Bi-Rads'].apply(lambda x: int(x[0]) > 3)
    xml_path = in_path / 'AllXML'
    dicom_path = in_path / 'AllDICOMs'

    dcm_files = list(dicom_path.iterdir())
    patient_paths = shuffled_patients_paths(dcm_files)


    len_ds = len(patient_paths)
    val_split = int(len_ds * (100 - val_size - test_size) / 100 )
    test_split = int(len_ds * (100 - test_size) / 100 )
    if split == 'train':
        indices = slice(0, val_split)
    elif split == 'trainval':
        indices = slice(0, test_split)
    elif split == 'val':
        indices = slice(val_split, test_split)
    else:
        assert split == 'test'
        indices = slice(test_split, None)

    dataset: list[tuple[np.ndarray, bool]] = []
    for patient in list(patient_paths)[indices]:
        for image_file in patient_paths[patient]:
            id_str = image_file.name.split('_')[0]
            xml_file = (xml_path / id_str).with_suffix('.xml')
            malignant = xls_sheet[xls_sheet['File Name'] == id_str]['malignant']
            label = tensorflow.keras.utils.to_categorical(malignant, num_classes=2)[0]
            image = read_image(image_file)
            for roi in extract_squares(image, xml_file):
                dataset.append((np.expand_dims(roi, -1), label))

    tf_dataset =  tensorflow.data.Dataset.from_generator(lambda: ShuffledDataset(dataset),
                                                        output_signature=(
                                                        tensorflow.TensorSpec(shape=(dim, dim, 1), dtype=np.float32),
                                                        tensorflow.TensorSpec(shape=(2, ), dtype=bool)
                                                    ))
    return tf_dataset.map(tf_flip_square)


def preprocess_inbreast_for_ganterfactual(split='trainval') -> tensorflow.data.Dataset:
    dotenv.load_dotenv()

    args = vars(parse_args())
    in_path = pathlib.Path(args['in'])
    test_size = float(args['test'])
    val_size = float(args['validation'])
    dim = int(args['dimension'])

    xls_sheet = pd.read_excel(in_path / 'INbreast.xls', sheet_name="Sheet1", dtype=str)[:-2]
    xls_sheet['malignant'] = xls_sheet['Bi-Rads'].apply(lambda x: int(x[0]) > 3)
    xml_path = in_path / 'AllXML'
    dicom_path = in_path / 'AllDICOMs'
    dcm_files = list(dicom_path.iterdir())
    patient_paths = shuffled_patients_paths(dcm_files)


    len_ds = len(patient_paths)
    val_split = int(len_ds * (100 - val_size - test_size) / 100 )
    test_split = int(len_ds * (100 - test_size) / 100 )
    if split == 'train':
        indices = slice(0, val_split)
    elif split == 'trainval':
        indices = slice(0, test_split)
    elif split == 'val':
        indices = slice(val_split, test_split)
    else:
        assert split == 'test'
        indices = slice(test_split, None)

    dataset_malignant: list[np.ndarray] = []
    dataset_benign: list[np.ndarray] = []
    for patient in list(patient_paths)[indices]:
        for image_file in patient_paths[patient]:
            if image_file.suffix == '.m':
                continue
            id_str = image_file.name.split('_')[0]
            xml_file = (xml_path / id_str).with_suffix('.xml')
            malignant = xls_sheet[xls_sheet['File Name'] == id_str]['malignant']
            image = read_image(image_file)
            for roi in extract_squares(image, xml_file):
                if malignant.item():
                    dataset_malignant.append(np.expand_dims(roi, -1))
                else:
                    dataset_benign.append(np.expand_dims(roi, -1))

    tf_dataset =  tensorflow.data.Dataset.from_generator(lambda: zip(ShuffledDataset(dataset_benign),
                                                                     ShuffledDataset(dataset_malignant)),
                                                        output_signature=(
                                                        tensorflow.TensorSpec(shape=(dim, dim, 1), dtype=np.float32),
                                                        tensorflow.TensorSpec(shape=(dim, dim, 1), dtype=np.float32),
                                                    ))
    return tf_dataset

