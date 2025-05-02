"""
Image segmentation
"""

import cv2

from methods import (
    thresholding,
    estimating_thresholding,
    intensity_rg_by,
    moments,
    all_methods,
    convolve_edges,
    convolve_edges_advanced,
)


class Segment:
    def __init__(self, segmentation_method):
        self.cam = cv2.VideoCapture(0)
        self._width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.method = segmentation_method

    def segment(self, **kwargs):
        while True:
            ret, frame = self.cam.read()
            if not ret:
                continue

            frame = self.method(frame, **kwargs)

            cv2.imshow("Processed", frame[:, ::-1])
            if cv2.waitKey(1) == ord('q'):
                break

        self.cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    segmenter = Segment(segmentation_method=convolve_edges_advanced)
    segmenter.segment()