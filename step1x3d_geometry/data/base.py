import math
import os
import json
import re
import cv2
from dataclasses import dataclass, field

import random
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from step1x3d_geometry.utils.typing import *


@dataclass
class BaseDataModuleConfig:
    root_dir: str = None
    data_dir: str = None
    npz_dir: str = None
    batch_size: int = 4
    num_workers: int = 8

    ################################# General argumentation #################################
    random_flip: bool = (
        False  # whether to randomly flip the input point cloud and the input images
    )

    ################################# Geometry part #################################
    load_geometry: bool = True  # whether to load geometry data
    with_sharp_data: bool = False
    geo_data_type: str = "sdf"  # occupancy, sdf
    # for occupancy or sdf supervision
    n_samples: int = 4096  # number of points in input point cloud
    upsample_ratio: int = 1  # upsample ratio for input point cloud
    sampling_strategy: Optional[str] = (
        "random"  # sampling strategy for input point cloud
    )
    scale: float = 1.0  # scale of the input point cloud and target supervision
    noise_sigma: float = 0.0  # noise level of the input point cloud
    rotate_points: bool = (
        False  # whether to rotate the input point cloud and the supervision, for VAE aug.
    )
    load_geometry_supervision: bool = False  # whether to load supervision
    supervision_type: str = "sdf"  # occupancy, sdf, tsdf, tsdf_w_surface
    n_supervision: int = 10000  # number of points in supervision
    tsdf_threshold: float = (
        0.01  # threshold for truncating sdf values, used when input is sdf
    )

    ################################# Image part #################################
    load_image: bool = False  # whether to load images
    load_aolp: bool = False
    image_type: str = "rgb"  # rgb, normal, rgb_or_normal
    image_file_type: str = "png"  # png, jpeg
    image_type_ratio: float = (
        1.0  # ratio of rgb for each dataset when image_type is "rgb_or_normal"
    )
    crop_image: bool = True  # whether to crop the input image
    random_color_jitter: bool = (
        False  # whether to randomly color jitter the input images
    )
    random_rotate: bool = (
        False  # whether to randomly rotate the input images, default [-10 deg, 10 deg]
    )
    random_mask: bool = False  # whether to add random mask to the input image
    background_color: Tuple[int, int, int] = field(
        default_factory=lambda: (255, 255, 255)
    )
    idx: Optional[List[int]] = None  # index of the image to load
    n_views: int = 1  # number of views
    foreground_ratio: Optional[float] = 0.90

    ################################# Caption part #################################
    load_caption: bool = False  # whether to load captions
    load_label: bool = False  # whether to load labels


class BaseDataset(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: BaseDataModuleConfig = cfg
        self.split = split

        self.uids = json.load(open(f"{cfg.root_dir}/{split}.json"))
        print(f"Loaded {len(self.uids)} {split} uids")

        # add ColorJitter transforms for input images
        if self.cfg.random_color_jitter:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
            )

        # add RandomRotation transforms for input images
        if self.cfg.random_rotate:
            self.rotate = transforms.RandomRotation(
                degrees=10, fill=(*self.cfg.background_color, 0.0)
            )  # by default 10 deg

    def __len__(self):
        return len(self.uids)

    def _load_shape_from_occupancy_or_sdf(self, index: int) -> Dict[str, Any]:
        if self.cfg.geo_data_type == "sdf":
            data = np.load(f"{self.cfg.npz_dir}/{self.uids[index]}/samples.npz")
            # for input point cloud
            surface = data["surface"]
            if self.cfg.with_sharp_data:
                sharp_surface = data["sharp_surface"]
        else:
            raise NotImplementedError(
                f"Data type {self.cfg.geo_data_type} not implemented"
            )

        # random sampling
        if self.cfg.sampling_strategy == "random":
            rng = np.random.default_rng()
            ind = rng.choice(
                surface.shape[0],
                self.cfg.upsample_ratio * self.cfg.n_samples,
                replace=True,
            )
            surface = surface[ind]
            if self.cfg.with_sharp_data:
                sharp_surface = sharp_surface[ind]
        elif self.cfg.sampling_strategy == "fps":
            import fpsample

            kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(
                surface[:, :3], self.cfg.n_samples, h=5
            )
            surface = surface[kdline_fps_samples_idx]
            if self.cfg.with_sharp_data:
                kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(
                    sharp_surface[:, :3], self.cfg.n_samples, h=5
                )
                sharp_surface = sharp_surface[kdline_fps_samples_idx]
        else:
            raise NotImplementedError(
                f"sampling strategy {self.cfg.sampling_strategy} not implemented"
            )

        # rescale data
        surface[:, :3] = surface[:, :3] * self.cfg.scale  # target scale
        if self.cfg.with_sharp_data:
            sharp_surface[:, :3] = sharp_surface[:, :3] * self.cfg.scale  # target scale
            ret = {
                "uid": self.uids[index].split("/")[-1],
                "surface": surface.astype(np.float32),
                "sharp_surface": sharp_surface.astype(np.float32),
            }
        else:
            ret = {
                "uid": self.uids[index].split("/")[-1],
                "surface": surface.astype(np.float32),
            }

        return ret

    def _load_shape_supervision_occupancy_or_sdf(self, index: int) -> Dict[str, Any]:
        # for supervision
        ret = {}
        if self.cfg.geo_data_type == "sdf":
            # data = np.load(f"{self.cfg.root_dir}/surfaces/{self.uids[index]}.npz")
            data = np.load(f"{self.cfg.npz_dir}/{self.uids[index]}/samples.npz")
            data = np.concatenate(
                [data["volume_rand_points"], data["near_surface_points"]], axis=0
            )
            rand_points, sdfs = data[:, :3], data[:, 3:]
        else:
            raise NotImplementedError(
                f"Data type {self.cfg.geo_data_type} not implemented"
            )

        # random sampling
        rng = np.random.default_rng()
        ind = rng.choice(rand_points.shape[0], self.cfg.n_supervision, replace=False)
        rand_points = rand_points[ind]
        rand_points = rand_points * self.cfg.scale
        ret["rand_points"] = rand_points.astype(np.float32)

        if self.cfg.geo_data_type == "sdf":
            if self.cfg.supervision_type == "sdf":
                ret["sdf"] = sdfs[ind].flatten().astype(np.float32)
            elif self.cfg.supervision_type == "occupancy":
                ret["occupancies"] = np.where(sdfs[ind].flatten() < 1e-3, 0, 1).astype(
                    np.float32
                )
            elif self.cfg.supervision_type == "tsdf":
                ret["sdf"] = (
                    sdfs[ind]
                    .flatten()
                    .astype(np.float32)
                    .clip(-self.cfg.tsdf_threshold, self.cfg.tsdf_threshold)
                    / self.cfg.tsdf_threshold
                )
            else:
                raise NotImplementedError(
                    f"Supervision type {self.cfg.supervision_type} not implemented"
                )

        return ret

    def _load_image(self, index: int) -> Dict[str, Any]:
        def _process_img(image, aolp = None, background_color=(255, 255, 255), foreground_ratio=0.9):
            alpha = image.getchannel("A")
            bbox = alpha.getbbox()
            background = Image.new("RGBA", image.size, (*background_color, 255))
            image = Image.alpha_composite(background, image)
            image = image.crop(alpha.getbbox())

            new_size = tuple(int(dim * foreground_ratio) for dim in image.size)
            resized_image = image.resize(new_size)
            padded_image = Image.new("RGBA", image.size, (*background_color, 255))
            paste_position = (
                (image.width - resized_image.width) // 2,
                (image.height - resized_image.height) // 2,
            )
            padded_image.paste(resized_image, paste_position)

            # Expand image to 1:1
            max_dim = max(padded_image.size)
            image = Image.new("RGBA", (max_dim, max_dim), (*background_color, 255))
            paste_position = (
                (max_dim - padded_image.width) // 2,
                (max_dim - padded_image.height) // 2,
            )
            image.paste(padded_image, paste_position)
            image = image.resize((512, 512))
            if aolp is not None:
                background = Image.new("RGBA", aolp.size, (*background_color, 255))
                aolp = Image.alpha_composite(background, aolp)  # 准备背景合成Aolp
                aolp = aolp.crop(bbox)
                resized_aolp = aolp.resize(new_size)
                padded_aolp = Image.new("RGBA", aolp.size, (*background_color, 255))
                padded_aolp.paste(resized_aolp, paste_position)  # 这里背景填充为白色
                aolp = Image.new("RGBA", (max_dim, max_dim), (*background_color, 255))
                aolp.paste(padded_aolp, paste_position)
                aolp = aolp.resize((512, 512))
            return image.convert("RGB"), alpha  # aolp.convert("RGB")

        ret = {}
        if self.cfg.image_type == "rgb" or self.cfg.image_type == "normal":
            assert (
                self.cfg.n_views == 1
            ), "Only single view is supported for single image"
            sel_idx = random.choice(self.cfg.idx)  # 随机选择一个视角，训练其视角矫正学习能力
            ret["sel_image_idx"] = sel_idx
            obj_dir = os.path.join(self.cfg.data_dir, self.uids[index])
            if self.cfg.image_type == "rgb":
                img_path = f"{obj_dir}/normals_000_{sel_idx}.{self.cfg.image_file_type}"
                aolp_path = f"{obj_dir}/aop_000_{sel_idx}.{self.cfg.image_file_type}"
            elif self.cfg.image_type == "normal":
                img_path = f"{obj_dir}/normals_000_{sel_idx}.{self.cfg.image_file_type}"
                aolp_path = None
            image = Image.open(img_path).copy()

            aolp = None
            if self.cfg.load_aolp:
                aolp = Image.open(aolp_path).copy()
            # add random color jitter
            if self.cfg.random_color_jitter:
                rgb = self.color_jitter(image.convert("RGB"))
                image = Image.merge("RGBA", (*rgb.split(), image.getchannel("A")))

            # add random rotation
            # if self.cfg.random_rotate:
            #     image = self.rotate(image)
            #     # image = self.rotate(image)
            #     if aolp is not None:
            #         aolp = transforms.functional.rotate(aolp, angle, fill= (*self.cfg.background_color, 255))

            # add crop
            if self.cfg.crop_image:
                background_color = (
                    torch.randint(0, 256, (3,))
                    if self.cfg.background_color is None
                    else torch.as_tensor(self.cfg.background_color)
                )
                image, alpha= _process_img(
                    image, aolp, background_color, self.cfg.foreground_ratio
                )
            else:
                alpha = image.getchannel("A")
                background = Image.new("RGBA", image.size, self.cfg.background_color)
                image = Image.alpha_composite(background, image).convert("RGB")
                if image.size != (512, 512):
                    image = image.resize((512, 512))
                    if aolp is not None:
                        aolp = aolp.resize((512, 512))
            ret["image"] = torch.from_numpy(np.array(image) / 255.0)
            ret["mask"] = torch.from_numpy(np.array(alpha) / 255.0).unsqueeze(0)
            # if aolp is not None:
            #     # (H, W) -> (H, W, 1) -> /255.0
            #     aolp_np = np.array(aolp)
            #     ret["aolp"] = torch.from_numpy(aolp_np / 255.0)
        else:
            raise NotImplementedError(
                f"Image type {self.cfg.image_type} not implemented"
            )

        return ret

    def _get_data(self, index):
        ret = {"uid": self.uids[index]}

        # random flip
        # flip = np.random.rand() < 0.5 if self.cfg.random_flip else False
        flip = False
        # load geometry
        if self.cfg.load_geometry:
            if self.cfg.geo_data_type == "occupancy" or self.cfg.geo_data_type == "sdf":
                # load shape
                ret = self._load_shape_from_occupancy_or_sdf(index)
                # load supervision for shape
                if self.cfg.load_geometry_supervision:
                    ret.update(self._load_shape_supervision_occupancy_or_sdf(index))
            else:
                raise NotImplementedError(
                    f"Geo data type {self.cfg.geo_data_type} not implemented"
                )

            if flip:  # random flip the input point cloud and the supervision
                for key in ret.keys():
                    if key in ["surface", "sharp_surface"]:  # N x (xyz + normal)
                        ret[key][:, 0] = -ret[key][:, 0]
                        ret[key][:, 3] = -ret[key][:, 3]
                    elif key in ["rand_points"]:
                        ret[key][:, 0] = -ret[key][:, 0]

        # load image
        if self.cfg.load_image:
            ret.update(self._load_image(index))
            if flip:  # random flip the input image
                for key in ret.keys():
                    if key in ["image"]:  # random flip the input image
                        ret[key] = torch.flip(ret[key], [2])
                    if key in ["mask"]:  # random flip the input image
                        ret[key] = torch.flip(ret[key], [2])

        # load caption
        meta = None
        if self.cfg.load_caption:
            # with open(f"{self.cfg.root_dir}/metas/{self.uids[index]}.json", "r") as f:
            #     meta = json.load(f)
            # ret.update({"caption": meta["caption"]})
            obj_dir = os.path.join(self.cfg.data_dir, self.uids[index])
            txt_file_path = f"{obj_dir}/caption.txt"
            with open(txt_file_path, "r", encoding="utf-8") as f:
                caption_str = f.read().strip()

            ret.update({"caption": caption_str})

        # load label
        if self.cfg.load_label:
            if meta is None:
                with open(
                    f"{self.cfg.root_dir}/metas/{self.uids[index]}.json", "r"
                ) as f:
                    meta = json.load(f)
            ret.update({"label": [meta["label"]]})

        return ret

    def __getitem__(self, index):
        # try:
        return self._get_data(index)
        # except Exception as e:
        #     print(f"Error in {self.uids[index]}: {e}")
        #     return self.__getitem__(np.random.randint(len(self)))

    def collate(self, batch):
        from torch.utils.data._utils.collate import default_collate_fn_map

        return torch.utils.data.default_collate(batch)
