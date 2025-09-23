import os
import napari
import mrcfile
import argparse
import starfile
import numpy as np


def parser_helper(description=None):
    description = "Show coordinates as points in denoised tomogram" if description is None else description
    parser = argparse.ArgumentParser(description, add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tomo_path', type=str, required=True,
                        help='path to the folder containing (denoised) tomograms')
    parser.add_argument('--filename', type=str, required=True,
                        help='Tomogram name (including the file extension')
    parser.add_argument('--star_file', type=str, required=True,
                        help='Path to the starfile')
    parser.add_argument('--point_size', type=str, required=False, default=15,
                        help='Size of the points to be plotted in napari')
    return parser


def main(tomo_path, filename, star_file, point_size):
    tomo = mrcfile.open(os.path.join(tomo_path, filename)).data
    points = starfile.read(star_file)
    if type(points) is dict:
        points = points['particles']
    points = points[points['rlnMicrographName'] == filename.split('.')[0]]

    v = napari.Viewer()
    v.add_image(tomo, name='tomo')

    if 'rlnClassNumber' in points:
        classes = np.unique(points['rlnClassNumber']).tolist()
        for c in classes:
            current_points = points[points['rlnClassNumber'] == c]
            print(current_points.shape)
            v.add_points(current_points[['rlnCoordinateZ', 'rlnCoordinateY', 'rlnCoordinateX']],
                         name=f'class_{c}_points',
                         n_dimensional=True, size=point_size)
    else:
        v.add_points(points[['rlnCoordinateZ', 'rlnCoordinateY', 'rlnCoordinateX']], name='points',
                     n_dimensional=True, size=point_size)
    napari.run()


if __name__ == '__main__':
    parser = parser_helper()
    args = parser.parse_args()
    main(args.tomo_path, args.filename, args.star_file, args.point_size)
