import os
import h5py
import yaml
import napari
import mrcfile
import argparse
import numpy as np


def parser_helper(description=None):
    description = "Plot instance segmentation with napari" if description is None else description
    parser = argparse.ArgumentParser(description, add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, required=True,
                        help='path to the config file used for running CryoSiam')
    parser.add_argument('--filename', type=str, required=True,
                        help='Tomogram filename (including the file extension')
    return parser


def main(config, filename):
    with open(config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
    tomo = mrcfile.open(os.path.join(config['data_folder'], filename)).data
    instances_file = os.path.join(config['prediction_folder'] + '_filtered',
                                  filename.split(config['file_extension'])[0] + '_instance_preds.h5')
    with h5py.File(instances_file, 'r') as f:
        instances = f['instances'][()]
    v = napari.Viewer()
    v.add_image(tomo * -1, name='tomo')
    v.add_labels(instances, name='instances')

    prediction_file = os.path.join(config['filtering_mask_folder'],
                                   filename.split(config['file_extension'])[0] + '_preds.h5')
    with h5py.File(prediction_file, 'r') as f:
        labels = f['labels'][()]

    v.add_labels(np.isin(labels, config['filtering_mask_labels']), name='mask')
    v.add_labels((instances > 0) * 2, name='filtered_particles')
    napari.run()


if __name__ == '__main__':
    parser = parser_helper()
    args = parser.parse_args()
    main(args.config, args.filename)
