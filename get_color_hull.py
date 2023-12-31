import cv2
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import click


@click.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--scale-factor", default=0.25, help="Scale factor")
@click.option("--L-min", default=0, help="Minimum L value")
@click.option("--L-max", default=255, help="Maximum L value")
@click.option("--dist-thresh", default=1.9, help="Distance threshold")
@click.option("--min-neighbors", default=4, help="Minimum neighbors")
def get_lab_ab_hull_cli(path, scale_factor, l_min, l_max, dist_thresh, min_neighbors):
    """
    This script processes an image at the given path and returns the area of the convex
    hull of the points that result when mapping its color information in the a*-b* plane
    in the Lab color space. The arguments l-min and l-max can be used to mask out pixels
    below or above a certain Luminance value. dist_tresh and min_neighbors are used to
    filter outliers in a*-b* plane (points in the a*-b* plane are distributed in a discret
    raster).

    Args:
    path (str): Path to the image file.
    path (str): Path to the image file.
    scale_factor (float): Scale factor of the input image, scaling down typically reduces color noise.
    l_min (int): Minimum Luminance value for a pixel to be taken into account.
    l_max (int): Maximum Luminance value for a pixel to be taken into account.
    dist_thresh (float): Within wich distance threshold shall be searched for neighbors.
    min_neighbors (int): Minimum neighbors that have to be found within threshold to keep point.
    """
    # Call the original function with provided arguments
    result = get_lab_ab_hull(
        path, scale_factor, l_min, l_max, dist_thresh, min_neighbors
    )

    # You can print or return the result as needed
    print(result)


def get_lab_ab_hull(path, scale_factor, L_min, L_max, dist_thresh, min_neighbors):
    image = cv2.imread(path)

    # Resize the image
    height, width = image.shape[:2]
    image = cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)))

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    a, b = cv2.split(lab)[1:]

    # mask by luminance
    lower = np.array([L_min, 0, 0])
    upper = np.array([L_max, 255, 255])
    mask = cv2.inRange(lab, lower, upper)
    a = a[mask.nonzero()]
    b = b[mask.nonzero()]

    ab = np.column_stack((a.flatten(), b.flatten()))
    # remove duplicates
    ab = np.unique(ab, axis=0)

    # Calculate pairwise distances between points
    distances = cdist(ab, ab)
    # Set diagonal elements (self distances) to a value larger than the distance threshold
    np.fill_diagonal(distances, np.inf)
    # mask by neighbors within distance
    neighbor_mask = np.sum(distances <= dist_thresh, axis=1) >= min_neighbors

    hull = ConvexHull(ab[neighbor_mask])
    return hull.area


if __name__ == "__main__":
    get_lab_ab_hull_cli()
