import cv2
import numpy as np

heatmap = None

def generate_heatmap(frame, points):

    global heatmap

    if heatmap is None:

        heatmap = np.zeros_like(
            frame[:, :, 0]
        ).astype(np.float32)

    for point in points:

        x, y = point

        cv2.circle(
            heatmap,
            (x, y),
            20,
            1,
            -1
        )

    heatmap_blur = cv2.GaussianBlur(
        heatmap,
        (51, 51),
        0
    )

    heatmap_normalized = cv2.normalize(
        heatmap_blur,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    heatmap_colored = cv2.applyColorMap(
        heatmap_normalized.astype(np.uint8),
        cv2.COLORMAP_JET
    )

    return heatmap_colored