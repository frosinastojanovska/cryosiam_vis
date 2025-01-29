import os
import h5py
import yaml
import napari
import mrcfile
import argparse


def parser_helper(description=None):
    description = "Plot instance segmentation with napari" if description is None else description
    parser = argparse.ArgumentParser(description, add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, required=True,
                        help='path to the config file used for running SimSiam')
    parser.add_argument('--tomo', type=str, required=True,
                        help='Tomogram name (including the file extension')
    return parser


def main(args):
    with open(args.config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
    tomo_name = args.tomo
    tomo = mrcfile.open(os.path.join(config['data_folder'], tomo_name)).data
    instances_file = os.path.join(config['instances_mask_folder'],
                                  tomo_name.split(config['file_extension'])[0] + '_instance_preds.h5')
    with h5py.File(instances_file, 'r') as f:
        instances = f['instances'][()]
    v = napari.Viewer()
    v.add_image(tomo, name='tomo')
    v.add_labels(instances, name='instances')


if __name__ == '__main__':
    parser = parser_helper()
    args = parser.parse_args()
    main(args)
    napari.run()
