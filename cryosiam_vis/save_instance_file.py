import os
import math
import h5py
import yaml
import mrcfile
import argparse
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import expand_labels


def save_tomogram(file_path, data):
    """Save a numpy array as tomogram in MRC or REC file format.
    :param file_path: path to the file
    :type file_path: str
    :param data: the data to be stored as tomogram
    :type data: np.array
    :return: tomogram as numpy array as confirmation of the saving
    :rtype: np.array
    """
    with mrcfile.new(file_path, data=data, overwrite=True) as m:
        return m.data


def generate_particle_subtomogram(tomo, instances, instance_id, masked):
    mask = instances == instance_id
    slices = ndi.find_objects(mask)[0]
    sub_mask = mask[slices]
    patch = tomo[slices].copy()
    if masked:
        patch[sub_mask != 1] = 0
    patch_size = [64, 64, 64]
    pad_size = tuple([(max(math.ceil((p - b) / 2), 0), max(math.floor((p - b) / 2), 0)) for p, b in
                      zip(patch_size, patch.shape)])
    patch = np.pad(patch, pad_size, 'constant')[:patch_size[0], :patch_size[1], :patch_size[2]]
    return patch


def parser_helper(description=None):
    description = "Plot instance segmentation with napari" if description is None else description
    parser = argparse.ArgumentParser(description, add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, required=True,
                        help='path to the config file used for running SimSiam')
    parser.add_argument('--tomo', type=str, required=True,
                        help='Tomogram name (including the file extension')
    parser.add_argument('--instance', type=int, required=True,
                        help='Instance id of the structure to be saved')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the output folder to save the instance subtomogram')
    parser.add_argument('--mask', action='store_true', required=False,
                        help='Instance id of the structure to be saved')
    return parser


def main(args):
    with open(args.config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
    tomo_name = args.tomo
    inst_id = args.instance
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    tomo = mrcfile.open(os.path.join(config['data_folder'], tomo_name)).data
    tomo_root_name = tomo_name.split(config['file_extension'])[0]
    instances_file = os.path.join(config['instances_mask_folder'], tomo_root_name + '_instance_preds.h5')
    with h5py.File(instances_file, 'r') as f:
        instances = f['instances'][()]

    # instances = expand_labels(instances, 1)

    subtomo = generate_particle_subtomogram(tomo, instances, inst_id, args.mask)
    save_tomogram(os.path.join(out_dir, f'{tomo_root_name}_instance_{inst_id}.mrc'), subtomo)


if __name__ == '__main__':
    parser = parser_helper()
    args = parser.parse_args()
    main(args)
