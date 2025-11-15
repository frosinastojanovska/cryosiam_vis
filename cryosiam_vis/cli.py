import argparse

from cryosiam_vis.visualize_semantic_segmentation import main as visualize_semantic_segmentation
from cryosiam_vis.visualize_denoised_tomogram import main as visualize_denoising
from cryosiam_vis.visualize_instance_segmentation import main as visualize_instance_segmentation
from cryosiam_vis.visualize_filtered_instance_segmentation import main as visualize_filtered_instance_segmentation
from cryosiam_vis.visualize_coordinates_from_star_file import main as visualize_coordinates_from_star_file
from cryosiam_vis.visualize_embeddings import main as visualize_embeddings
from cryosiam_vis.visualize_clusters import main as visualize_embeddings_clusters

__version__ = "1.0"


def main():
    parser = argparse.ArgumentParser(prog="cryosiam-vis", description="CryoSiam Vis Command Line Interface")
    parser.add_argument("--version", action="version", version=f"CryoSiam-Vis {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # visualize_denoising subcommand
    sp_denoise = subparsers.add_parser("visualize_denoising", help="Run visualization of the denoising prediction")
    sp_denoise.add_argument('--config_file', type=str, required=True,
                            help='Path to the .yaml configuration file that was used while running CryoSiam denoise_predict command')
    sp_denoise.add_argument('--filename', type=str, required=True,
                            help='The filename of the tomogram to be visualized')
    sp_denoise.set_defaults(func=lambda args: visualize_denoising(args.config_file, args.filename))

    # visualize_semantic subcommand
    sp_semantic = subparsers.add_parser("visualize_semantic",
                                        help="Run visualization of the sematic segmentation prediction")
    sp_semantic.add_argument('--config_file', type=str, required=True,
                             help='Path to the .yaml configuration file that was used while running CryoSiam semantic_predict command')
    sp_semantic.add_argument('--filename', type=str, required=True,
                             help='The filename of the tomogram to be visualized')
    sp_semantic.set_defaults(func=lambda args: visualize_semantic_segmentation(args.config_file, args.filename))

    # visualize_instance subcommand
    sp_instance = subparsers.add_parser("visualize_instance",
                                        help="Run visualization of the instance segmentation prediction")
    sp_instance.add_argument('--config_file', type=str, required=True,
                             help='Path to the .yaml configuration file that was used while running CryoSiam instance_predict command')
    sp_instance.add_argument('--filename', type=str, required=True,
                             help='The filename of the tomogram to be visualized')
    sp_instance.set_defaults(func=lambda args: visualize_instance_segmentation(args.config_file, args.filename))

    #visualize_filtered_instances subcommand
    sp_instance = subparsers.add_parser("visualize_filtered_instance",
                                        help="Run visualization of the instance segmentation prediction")
    sp_instance.add_argument('--config_file', type=str, required=True,
                             help='Path to the .yaml configuration file that was used while running CryoSiam instance_filter command')
    sp_instance.add_argument('--filename', type=str, required=True,
                             help='The filename of the tomogram to be visualized')
    sp_instance.set_defaults(func=lambda args: visualize_filtered_instance_segmentation(args.config_file, args.filename))

    # visualize_coordinates
    sp_coordinates = subparsers.add_parser("visualize_coordinates",
                                           help="Run visualization of the instance segmentation prediction")
    sp_coordinates.add_argument('--config_file', type=str, required=True,
                        help='path to the config file used for running CryoSiam semantic particle identification')
    sp_coordinates.add_argument('--filename', type=str, required=True,
                        help='Tomogram filename (including the file extension')
    sp_coordinates.add_argument('--point_size', type=str, required=False, default=15,
                        help='Size of the points to be plotted in napari')
    sp_coordinates.set_defaults(
        func=lambda args: visualize_coordinates_from_star_file(args.config_file, args.filename, args.point_size))

    # visualize_embeddings
    sp_embeddings = subparsers.add_parser("visualize_embeddings",
                                          help="Run visualization of the predicted embeddings")
    sp_embeddings.add_argument('--config_file', type=str, required=True,
                             help='Path to the .yaml configuration file that was used while running CryoSiam simsiam_visualize_embeddings command')
    sp_embeddings.set_defaults(func=lambda args: visualize_embeddings(args.config_file))

    # visualize_embeddings_clusters
    sp_embeddings = subparsers.add_parser("visualize_embeddings_clusters",
                                          help="Run visualization of the predicted embeddings with clusters")
    sp_embeddings.add_argument('--config_file', type=str, required=True,
                               help='Path to the .yaml configuration file that was used while running CryoSiam cluster embeddings command')
    sp_embeddings.add_argument('--clustering', type=str, required=True,
                               help='clustering type, one of ["kmeans", "spectral"]')

    sp_embeddings.set_defaults(func=lambda args: visualize_embeddings_clusters(args.config_file, args.clustering))

    args = parser.parse_args()
    # Run selected command
    args.func(args)
