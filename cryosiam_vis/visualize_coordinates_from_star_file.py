import os
import yaml
import napari
import mrcfile
import argparse
import starfile
import numpy as np


def parser_helper(description=None):
    description = "Show coordinates as points in denoised tomogram" if description is None else description
    parser = argparse.ArgumentParser(description, add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config_file', type=str, required=True,
                        help='path to the config file used for running CryoSiam semantic particle identification')
    parser.add_argument('--filename', type=str, required=True,
                        help='Tomogram filename (including the file extension')
    parser.add_argument('--point_size', type=str, required=False, default=15,
                        help='Size of the points to be plotted in napari')
    return parser


def main(config, filename, point_size):
    with open(config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
    tomo = mrcfile.open(os.path.join(config['data_folder'], filename)).data
    labels_files = [x for x in os.listdir(config['prediction_folder']) if x.endswith('_particles.star')]

    v = napari.Viewer()
    v.add_image(tomo * -1, name='tomo')

    for label_file in labels_files:
        points = starfile.read(os.path.join(config['prediction_folder'], label_file))
        if type(points) is dict:
            points = points['particles']
        if points.shape[0] == 0:
            continue
        points = points[points['rlnMicrographName'] == filename.split('.')[0]]
        v.add_points(points[['rlnCoordinateZ', 'rlnCoordinateY', 'rlnCoordinateX']],
                     name=label_file.split('.star')[0],
                     n_dimensional=True, size=point_size)
    napari.run()


if __name__ == '__main__':
    parser = parser_helper()
    args = parser.parse_args()
    main(args.config_file, args.filename, args.point_size)
