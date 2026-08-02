"""
processing_prismatic.py

HuggingFace-style preprocessor for Prismatic VLMs. Self-contained.
"""

from typing import ClassVar, List, Optional, Tuple, Union

import timm.data
import torch
import torchvision.transforms.functional as TVF
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import PreTrainedTokenizerBase
from transformers.image_processing_utils import BatchFeature, ImageProcessingMixin
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import TensorType


def letterbox_pad_transform(image: Image.Image, padding_fill_value: Tuple[int, int, int]) -> Image.Image:
    """Pad PIL image to square by adding a symmetric border."""
    (w, h), max_wh = image.size, max(image.size)
    horizontal_pad, vertical_pad = int((max_wh - w) / 2), int((max_wh - h) / 2)
    padding = (horizontal_pad, vertical_pad, horizontal_pad, vertical_pad)
    return TVF.pad(image, padding, fill=padding_fill_value, padding_mode="constant")


class PrismaticImageProcessor(ImageProcessingMixin):
    model_input_names: ClassVar[List[str]] = ["pixel_values"]

    def __init__(
        self,
        use_fused_vision_backbone: bool = False,
        image_resize_strategy: str = "letterbox",
        input_sizes: Optional[List[Tuple[int, int, int]]] = None,
        interpolations: Optional[List[str]] = None,
        means: Optional[List[Tuple[float, float, float]]] = None,
        stds: Optional[List[Tuple[float, float, float]]] = None,
        **kwargs: str,
    ) -> None:
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.image_resize_strategy = image_resize_strategy

        input_sizes = [(3, 224, 224)] if input_sizes is None else input_sizes
        means = [(0.5, 0.5, 0.5)] if means is None else means
        stds = [(0.5, 0.5, 0.5)] if stds is None else stds

        self.input_sizes, self.interpolations, self.means, self.stds = input_sizes, interpolations, means, stds
        self.tvf_resize_params, self.tvf_crop_params, self.tvf_normalize_params = [], [], []
        self.tvf_do_letterbox, self.tvf_letterbox_fill = False, None

        for idx in range(len(input_sizes)):
            transform = timm.data.create_transform(
                input_size=self.input_sizes[idx],
                interpolation=self.interpolations[idx] if self.interpolations else "bicubic",
                mean=self.means[idx],
                std=self.stds[idx],
                crop_pct=1.0,
                crop_mode="center",
                is_training=False,
            )
            if not (
                isinstance(transform, Compose)
                and (len(transform.transforms) == 4)
                and isinstance(transform.transforms[0], Resize)
                and isinstance(transform.transforms[1], CenterCrop)
                and isinstance(transform.transforms[2], ToTensor)
                and isinstance(transform.transforms[3], Normalize)
            ):
                raise ValueError(f"Unexpected TIMM image transformation structure: `{transform}`")

            resize_t, crop_t, norm_t = transform.transforms[0], transform.transforms[1], transform.transforms[3]
            self.tvf_resize_params.append({
                "size": resize_t.size, "interpolation": TVF.pil_modes_mapping[resize_t.interpolation],
                "max_size": None, "antialias": True,
            })
            self.tvf_crop_params.append({"output_size": crop_t.size})
            self.tvf_normalize_params.append({
                "mean": norm_t.mean.float().numpy().tolist(),
                "std": norm_t.std.float().numpy().tolist(),
                "inplace": False,
            })
            self.tvf_do_letterbox, self.tvf_letterbox_fill = False, None

            if self.image_resize_strategy == "resize-naive":
                self.tvf_resize_params[idx]["size"] = (resize_t.size, resize_t.size)
            elif self.image_resize_strategy == "letterbox":
                self.tvf_do_letterbox, self.tvf_letterbox_fill = True, tuple([int(x * 255) for x in self.means[idx]])
            elif self.image_resize_strategy == "resize-crop":
                pass
            else:
                raise ValueError(f"Image resize strategy `{self.image_resize_strategy}` is not supported!")

        super().__init__(**kwargs)

    def apply_transform(self, img: Image.Image) -> torch.Tensor:
        if self.tvf_do_letterbox:
            img = letterbox_pad_transform(img, self.tvf_letterbox_fill)

        imgs_t = []
        for idx in range(len(self.input_sizes)):
            img_idx = TVF.resize(img, **self.tvf_resize_params[idx])
            img_idx = TVF.center_crop(img_idx, **self.tvf_crop_params[idx])
            img_idx_t = TVF.to_tensor(img_idx)
            img_idx_t = TVF.normalize(img_idx_t, **self.tvf_normalize_params[idx])
            imgs_t.append(img_idx_t)

        return torch.vstack(imgs_t)

    def preprocess(
        self,
        images: Union[Image.Image, List[Image.Image]],
        return_tensors: Optional[Union[str, TensorType]] = None,
        **_: str,
    ) -> BatchFeature:
        if not isinstance(images, list):
            images = [images]
        pixel_values = torch.stack([self.apply_transform(img.convert("RGB")) for img in images])
        return BatchFeature(data={"pixel_values": pixel_values.float().numpy()}, tensor_type=return_tensors)

    def __call__(self, images: Union[Image.Image, List[Image.Image]], **kwargs) -> BatchFeature:
        return self.preprocess(images, **kwargs)


class PrismaticProcessor(ProcessorMixin):
    attributes: ClassVar[List[str]] = ["image_processor", "tokenizer"]
    image_processor_class: str = "AutoImageProcessor"
    tokenizer_class: str = "AutoTokenizer"

    def __init__(
        self,
        image_processor: Optional[ImageProcessingMixin] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> None:
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        images: Union[Image.Image, List[Image.Image]],
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Optional[Union[bool, str, TruncationStrategy]] = None,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        pixel_values = self.image_processor(images, return_tensors=return_tensors)["pixel_values"]
        text_inputs = self.tokenizer(
            text, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length
        )
        if pixel_values.shape[0] != text_inputs.input_ids.shape[0]:
            if text_inputs.input_ids.shape[0] == 1:
                text_inputs.input_ids = text_inputs.input_ids.repeat(pixel_values.shape[0], 1)
                text_inputs.attention_mask = text_inputs.attention_mask.repeat(pixel_values.shape[0], 1)
            else:
                raise ValueError("Batch is malformed; expected same number of images and text inputs!")
        return BatchFeature(data={**text_inputs, "pixel_values": pixel_values})

    def batch_decode(self, sequences, skip_special_tokens=False, clean_up_tokenization_spaces=None, **kwargs):
        return self.tokenizer.batch_decode(
            sequences, skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces, **kwargs,
        )

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=None, **kwargs):
        return self.tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces, **kwargs,
        )

    @property
    def model_input_names(self) -> List[str]:
        return list(dict.fromkeys(self.tokenizer.model_input_names + self.image_processor.model_input_names))
