import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from typing import Dict, List, Optional, Union, Tuple, Iterable

ImgNet_std_mean = [0.5, 0.5, 0.5]
ImgNet_std_std = [0.5, 0.5, 0.5]


def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n" # \n is seperator token


def resize(image: Image, size: Tuple[int, int],
           resample: Image.Resampling=None,
           reducing_gap: Optional[int]=None) -> np.ndarray:
    height, width = size
    resized_img = image.resize(
        (width, height), resample=resample, reducing_gap=reducing_gap
    )
    return resized_img


def normalize(image: np.ndarray,
              mean: Union[float, Iterable[float]],
              std: Union[float, Iterable[float]]) -> np.ndarray:
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image


def rescale(image: np.ndarray, scale: float, dtype: np.dtype=np.float32) -> np.ndarray:
    rescaled_img = image * scale # because scale factor is 1/255
    rescaled_img = rescaled_img.astype(dtype)
    return rescaled_img

def process_images(images: List[Image.Image],
                   size: Dict[str, int]=None,
                   resample: Image.Resampling=None,
                   rescale_factor: float=None,
                   img_mean: Optional[Union[float, List[float]]]=None,
                   img_std: Optional[Union[float, List[float]]]=None) -> List[np.ndarray]:
    height, width = size[0], size[1]
    images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]
    images = [np.array(image) for image in images]

    # rescale the pixel values in range [0, 1]
    images = [rescale(image, scale=rescale_factor) for image in images]
    images = [normalize(image, mean=img_mean, std=img_std) for image in images]

    # changing to (c, h, w)
    images = [image.transpose(2, 0, 1) for image in images]
    return images


class paligemma_proc:
    # these are placeholder tokens which will then be replaced by the vision transformer tokens
    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()
        self.image_seq_len = num_image_tokens
        self.image_size = image_size

        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]  # these tokens are used for obj detection (bounding boxes)
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ]  # these are for obj segmentation
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(self, text: List[str], images: List[Image.Image],
                 padding: str="longest", truncation: bool=True) -> dict:
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} for {len(text)} prompts"

        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size), # resizing to given size for input
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1/255.0,
            image_mean=ImgNet_std_mean,
            image_std=ImgNet_std_std,
        )

        # convert list of numpy arrays to a single numpy array with shape (bs, c, h, w)
        pixel_values = np.stack(pixel_values, axis=0)
        pixel_values = torch.tensor(pixel_values)

        input_strings = [
            add_image_tokens_to_prompt(
            bos_token = self.tokenizer.bos_token,
            image_seq_len = self.image_seq_len,
            image_token = self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        # return inputA_ids and attn_mask as torch.tensors
        # eg:- fc barcelona --> [1, 3, 6] --> [[....], [....], [....]] (each of 1024 dim)
        inputs = self.tokenizer(input_strings,
                                return_tensors="pt",
                                padding=padding,
                                truncation=truncation,
                                )

        return_data = {"pixel_values": pixel_values, **inputs}
        return return_data


