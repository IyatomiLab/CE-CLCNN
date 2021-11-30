import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchtyping import TensorType


def visualize_char_img(
    save_dir: str,
    char_imgs_batch: TensorType["batch_size", "char_len", "char_w", "char_h"],  # type: ignore # NOQA: F821)
) -> None:

    for i in range(len(char_imgs_batch)):

        char_imgs = char_imgs_batch[i]
        char_imgs = char_imgs.detach().cpu().numpy()

        num_chars = len(char_imgs)
        num_chars = num_chars if num_chars < 10 else 10
        fig, axes = plt.subplots(ncols=num_chars, dpi=300)

        for j in range(num_chars):
            char_img = char_imgs[j]
            img = Image.fromarray(np.uint8(char_img * 255))
            axes[j].imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="none")
            axes[j].axis("off")

        png_file = os.path.join(save_dir, f"vis_char_{i:03d}.png")
        fig.savefig(png_file, bbox_inches="tight")
        plt.close()
