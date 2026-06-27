"""Color attribute extraction.

Dominant colors are extracted with K-means clustering (HSV/RGB space) as
described in the CAPSTONE paper (Section 3, "Visual Perception Pipeline").
For each detected object we cluster its cropped pixels into ``k`` groups and
report each centroid together with the proportion of pixels assigned to it.
"""

from collections import Counter

import numpy as np
from sklearn.cluster import KMeans


def rgb_to_hex(rgb):
    """Convert an ``(r, g, b)`` tuple/array to a ``#rrggbb`` hex string."""
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def get_dominant_colors(image, k=3):
    """Extract the ``k`` dominant colors of an image crop via K-means.

    Parameters
    ----------
    image : np.ndarray
        ``(H, W, 3)`` RGB image (or crop).
    k : int
        Number of color clusters (paper uses K = 3).

    Returns
    -------
    dict
        Mapping ``{hex_color: "xx.x%"}`` describing the color distribution.
    """
    pixels = image.reshape(-1, 3)
    # Guard against crops with fewer pixels than requested clusters.
    n_clusters = min(k, max(1, len(np.unique(pixels, axis=0))))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    labels = kmeans.fit_predict(pixels)

    label_counts = Counter(labels)
    total_count = sum(label_counts.values())
    colors = kmeans.cluster_centers_.astype(int)

    return {
        rgb_to_hex(tuple(colors[i])): f"{round(label_counts[i] / total_count, 2) * 100}%"
        for i in label_counts
    }


def analyze_colors(image, num_colors=3):
    """Return only the single most dominant color (RGB + hex).

    A lighter-weight variant of :func:`get_dominant_colors` kept for
    convenience / scene-level summaries.
    """
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors, n_init=10, random_state=42).fit(pixels)
    counts = Counter(kmeans.labels_)
    dominant = kmeans.cluster_centers_[counts.most_common(1)[0][0]].astype(int)

    return {
        "dominant_rgb": tuple(int(x) for x in dominant),
        "dominant_hex": rgb_to_hex(dominant),
    }
