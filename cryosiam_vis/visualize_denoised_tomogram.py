import os
import yaml
import napari
import mrcfile
import argparse


def parser_helper(description=None):
    description = "Show semantic segmentation with napari" if description is None else description
    parser = argparse.ArgumentParser(description, add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, required=True,
                        help='path to the config file used for running CryoSiam denoising')
    parser.add_argument('--filename', type=str, required=True,
                        help='Tomogram filename (including the file extension')
    return parser


def main(config, filename):
    with open(config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
    tomo = mrcfile.open(os.path.join(config['data_folder'], filename)).data
    denoised_tomo = mrcfile.open(os.path.join(config['prediction_folder'], filename)).data

    v = napari.Viewer()
    v.add_image(tomo * -1, name='tomo')
    v.add_image(denoised_tomo * -1, name='denoised_tomo')
    napari.run()


if __name__ == '__main__':
    parser = parser_helper()
    args = parser.parse_args()
    main(args.config, args.filename)
