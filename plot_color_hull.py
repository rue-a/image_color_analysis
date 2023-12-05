import cv2
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import click


@click.command()
@click.option(
    "-O",
    "--output-file",
    default="out.png",
    help="Output filename",
    type=click.Path(dir_okay=False, writable=True, exists=False, allow_dash=True),
)
@click.argument("path", type=click.Path(exists=True))
@click.option("--scale-factor", default=0.25, help="Scale factor")
@click.option("--l-min", default=0, help="Minimum L value")
@click.option("--l-max", default=255, help="Maximum L value")
@click.option("--dist-thresh", default=1.9, help="Distance threshold")
@click.option("--min-neighbors", default=4, help="Minimum neighbors")
@click.option("--draw-hull", default=True, help="draw hull")
@click.option("--point-size", default=1.0, help="size of points")
@click.option("--jitter", default=1.0, help="max jitter dist")
@click.option("--transparency", default=0.3, help="transparency of points")
def plot_lab_ab_hull_cli(
    output_file,
    path,
    scale_factor,
    l_min,
    l_max,
    dist_thresh,
    min_neighbors,
    draw_hull,
    point_size,
    jitter,
    transparency,
):
    """
    This script processes an image at the given path and plots its color information in the a*-b*
    plane of the Lab color space.

    Args:
    path (str): Path to the image file.
    scale_factor (float): Scale factor of the input image, scaling down typically reduces color noise.
    l_min (int): Minimum Luminance value for a pixel to be taken into account.
    l_max (int): Maximum Luminance value for a pixel to be taken into account.
    dist_thresh (float): Within wich distance threshold shall be searched for neighbors.
    min_neighbors (int): Minimum neighbors that have to be found within threshold to keep point.
    draw-hull (bool): Draw convex hull?
    point-size (float): Size of points.
    jitter (float): Maximum jitter distance.
    transparency (float): Transparency of points [0-1].
    """
    # Call the original function with provided arguments
    plot_lab_ab_hull(
        output_file,
        path,
        scale_factor,
        l_min,
        l_max,
        dist_thresh,
        min_neighbors,
        draw_hull,
        point_size,
        jitter,
        transparency,
    )


def lab_to_rgb(L, a, b):
    # Create a 1x1 LAB array
    lab_color = np.array([L, a, b], dtype=np.uint8).reshape((1, 1, 3))

    # Convert LAB to BGR using cv2
    bgr_color = cv2.cvtColor(lab_color, cv2.COLOR_LAB2BGR)

    # Convert BGR to RGB
    rgb_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2RGB)

    # Normalize RGB values to the range [0, 1]
    rgb_color_normalized = rgb_color / 255.0

    return rgb_color_normalized[0, 0, :]


def plot_lab_ab_hull(
    filename,
    path,
    scale_factor,
    L_min,
    L_max,
    dist_thresh,
    min_neighbors,
    draw_hull,
    point_size,
    jitter,
    transparency,
):
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

    a_flat = a.flatten()
    b_flat = b.flatten()
    ab = np.column_stack((a_flat, b_flat))
    # remove duplicates
    ab = np.unique(ab, axis=0)

    # Calculate pairwise distances between points
    distances = cdist(ab, ab)
    # Set diagonal elements (self distances) to a value larger than the distance threshold
    np.fill_diagonal(distances, np.inf)
    # mask by neighbors within distance
    neighbor_mask = np.sum(distances <= dist_thresh, axis=1) >= min_neighbors

    ab_masked = ab[neighbor_mask]
    hull = ConvexHull(ab_masked)

    jitter_amount = jitter  # Adjust as needed
    a_jittered = a_flat + np.random.uniform(
        low=-jitter_amount, high=jitter_amount, size=a_flat.shape
    )
    b_jittered = b_flat + np.random.uniform(
        low=-jitter_amount, high=jitter_amount, size=b_flat.shape
    )

    # get RBG color from a* and b* with fixed L
    L = 138
    rgb_colors = [lab_to_rgb(L, a_flat[i], b_flat[i]) for i in range(len(a_flat))]

    plt.figure(figsize=(8, 8))
    plt.scatter(
        a_jittered, b_jittered, c=rgb_colors, s=point_size, lw=0, alpha=transparency
    )
    plt.xlim(40, 220)
    plt.ylim(40, 220)
    plt.xlabel("a*")
    plt.ylabel("b*")

    # plot hull
    if draw_hull:
        for simplex in hull.simplices:
            plt.plot(ab_masked[simplex, 0], ab_masked[simplex, 1], "k-", lw=0.5)

        # write out hull area
        plt.text(
            plt.xlim()[1] - 3,
            plt.ylim()[0] + 3,
            f"hull area: {hull.area:.2f}",
            color="black",
            ha="right",
            va="bottom",
        )

    plt.title(filename)
    # plt.show()
    plt.savefig(filename, dpi=300)
    plt.close()


if __name__ == "__main__":
    plot_lab_ab_hull_cli()
