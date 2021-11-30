from functools import lru_cache
from typing import List

import numpy as np
from allennlp.common.from_params import FromParams
from PIL import Image, ImageDraw, ImageFont


class CharacterImageProcessor(FromParams):
    def __init__(self, font_name: str, font_size: int) -> None:
        super().__init__()
        self.font = ImageFont.truetype(
            font=font_name, size=int(font_size * 0.85), encoding="utf-8"
        )
        self.font_size = font_size

    def __call__(self, chars: List[str], is_normalize: bool = True) -> np.ndarray:
        char_img_list = [[self.process(char)] for char in chars]

        # shape: (char_len, font_size, font_size)
        char_imgs = np.vstack(char_img_list)

        # convert dtype to np.float32
        char_imgs = char_imgs.astype(np.float32)

        if is_normalize:
            # normalize to [0, 1]
            return char_imgs / 255
        else:
            return char_imgs

    @lru_cache(maxsize=5000)
    def process(self, c: str) -> np.ndarray:
        img_pil = self.char_to_font_img(c)

        try:
            img_np = self.resize_font_img(img_pil)
        except ValueError:
            img_np = np.zeros((self.font_size, self.font_size))

        return img_np

    def char_to_font_img(self, char: str) -> Image:
        img_size = np.ceil(np.array(self.font.getsize(char)) * 1.1).astype(int)

        img = Image.new("L", tuple(img_size), "black")
        draw = ImageDraw.Draw(img)
        text_offset = (img_size - self.font.getsize(char)) // 2
        draw.text(text_offset, char, font=self.font, fill="#fff")

        return img

    def resize_font_img(self, img: Image) -> np.ndarray:
        arr = np.asarray(img)
        r, c = np.where(arr != 0)
        r.sort()
        c.sort()

        if len(r) == 0:
            b = np.zeros((self.font_size, self.font_size))
        else:
            top = r[0]
            bottom = r[-1]
            left = c[0]
            right = c[-1]

            # trim character
            c_arr = arr[top:bottom, left:right]
            b = np.zeros((self.font_size, self.font_size), dtype=c_arr.dtype)
            r_offset = int((b.shape[0] - c_arr.shape[0]) / 2)
            c_offset = int((b.shape[1] - c_arr.shape[1]) / 2)
            b[
                r_offset : r_offset + c_arr.shape[0],
                c_offset : c_offset + c_arr.shape[1],
            ] = c_arr

        return b
