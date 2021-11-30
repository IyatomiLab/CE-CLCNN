import math
import random

import numpy as np
from allennlp.common.from_params import FromParams


class RandomErasing(FromParams):
    def __init__(
        self,
        p: float,
        max_area_ratio: float,
        min_area_ratio: float,
        max_aspect_ratio: float,
        min_aspect_ratio: float,
    ) -> None:
        super().__init__()

        self._p = p

        self._max_area_ratio = max_area_ratio
        self._min_area_ratio = min_area_ratio

        self._max_aspect_ratio = max_aspect_ratio
        self._min_aspect_ratio = min_aspect_ratio

    @property
    def s_l(self) -> float:
        return self._min_area_ratio

    @property
    def s_h(self) -> float:
        return self._max_area_ratio

    @property
    def r1(self) -> float:
        return self._min_aspect_ratio

    @property
    def r2(self) -> float:
        return self._max_aspect_ratio

    def apply_augmentations(self, xs: np.ndarray) -> np.ndarray:
        xs_augmented = [[self.apply_augmentation(xs[i])] for i in range(len(xs))]
        # shape: (char_len, char_w, char_h)
        return np.vstack(xs_augmented).astype(np.float32)

    def apply_augmentation(self, x: np.ndarray) -> np.ndarray:
        W, H = x.shape
        p1 = random.random()
        if p1 >= self._p:
            return x
        else:
            while True:
                S_e = random.uniform(self.s_l, self.s_h) * (W * H)
                r_e = random.uniform(self.r1, self.r2)
                H_e = int(math.sqrt(S_e * r_e))
                W_e = int(math.sqrt(S_e / r_e))

                x_e = random.randint(0, W)
                y_e = random.randint(0, H)
                if ((x_e + W_e) <= W) and (y_e + H_e) <= H:
                    top = x_e
                    bottom = x_e + W_e
                    left = y_e
                    right = y_e + H_e

                    noise = np.random.randint(0, 255, (bottom - top, right - left))
                    x[top:bottom, left:right] = noise

                    return x
