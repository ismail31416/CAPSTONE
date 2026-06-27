"""Geometric / spatial encoding helpers.

These functions turn raw bounding-box geometry into the discrete, symbolic
cues (quadrant, confidence band) that CAPSTONE feeds to the LLM.
"""


def get_quadrant(x, y, width, height):
    """Return the image quadrant (e.g. ``"top-left"``) for a point."""
    h_pos = "left" if x < width / 2 else "right"
    v_pos = "top" if y < height / 2 else "bottom"
    return f"{v_pos}-{h_pos}"


def confidence_label(conf):
    """Map a numeric confidence in ``[0, 1]`` to a human-readable band."""
    return (
        "very high" if conf >= 0.9
        else "high" if conf >= 0.75
        else "medium" if conf >= 0.5
        else "low" if conf >= 0.25
        else "very low"
    )
