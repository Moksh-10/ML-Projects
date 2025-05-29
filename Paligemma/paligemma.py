from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

IMAGENET_STD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD_STD = [0.5, 0.5, 0.5]


def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


def rescale(image: np.ndarray, scale: float, dtype: np.dtype = np.float32) -> np.ndarray:
    rescaled_img = image * scale
    rescaled_img = rescaled_img.astype(dtype)
    return rescaled_img


def resize(image: Image, size: Tuple[int, int], resample: Image.Resampling = None, reducing_gap: Optional[int] = None) -> np.ndarray:
    h, w = size
    resized_img = image.resize(
        (w, h), resample = resample, reducing_gap = reducing_gap
    )
    return resized_img


def normalize(image:np.ndarray, mean: Union[float, Iterable[float]], std: Union[float, Iterable[float]]) -> np.ndarray:
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image


def process_images(
        images: List[Image.Image],
        size: Dict[str, int] = None,
        resample: Image.Resampling = None,
        rescale_factor: float = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None
) -> List[np.ndarray]:
    h, w = size[0], size[1]
    images = [
        resize(image=img, size=(h, w), resample=resample) for img in images
    ]
    images = [np.array(img) for img in images]
    images = [rescale(img, scale=rescale_factor) for img in images]
    images = [normalize(img, mean=image_mean, std=image_std) for img in images]
    images = [image.transpose(2, 0, 1) for image in images]
    return images


class PaliGemmaProcessor:
    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()
        self.image_sl = num_image_tokens
        self.img_size = image_size

        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [f"<loc{i:04d}>" for i in range(1024)]
        EXTRA_TOKENS += [f"<seg{i:03d}>" for i in range(128)]
        tokenizer.add_tokens(EXTRA_TOKENS)
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokennizer = tokenizer

    def __call__(self, text: List[str], images: List[Image.Image],
                 padding: str = "longest",
                 truncation: bool = True) -> dict:
        assert len(images) == 1 and len(text) == 1, f"received {len(images)} for {len(text)} prompts."

        pixel_value = process_images(images,
                                     size=(self.img_size, self.img_size),
                                     resample=Image.Resampling.BICUBIC,
                                     rescale_factor=1/255.0,
                                     image_mean=IMAGENET_STD_MEAN,
                                     image_std=IMAGENET_STD_STD)
        pixel_value = np.stack(pixel_value, axis=0)
        pixel_value = torch.tensor(pixel_value)

        input_str = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokennizer.bos_token,
                image_seq_len=self.image_sl,
                image_token=self.IMAGE_TOKEN
            )
            for prompt in text
        ]

        inputs = self.tokennizer(
            input_str,
            return_tensor="pt",
            padding=padding,
            truncation=truncation
        )

        return_data = {"pixel_values": pixel_value, **inputs}
        return return_data


