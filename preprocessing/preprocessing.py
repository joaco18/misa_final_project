import logging
import numpy as np
from pathlib import Path
import SimpleITK as sitk
from scipy.ndimage import zoom
from skimage.exposure import match_histograms

logging.basicConfig(level=logging.INFO, format='%(message)s')

DEFAULT_NORM_CONFIG = {
    'type': 'min_max',
    'max_val': 255,
    'mask': None,
    'percentiles': (1, 99),
    'dtype': np.uint8
}

DEFAULT_HIST_MATCH_CONFIG = {
    'ref_img_path': '/home/jseia/Desktop/MAIA/classes/spain/misa/final_project/\
        misa_final_project/data/train_set/IBSR_18/IBSR_18_n4.nii.gz'
}

DEFAULT_RESIZE_CONFIG = {
    'voxel_size': (1, 1, 1),
    'interpolation_order': 3,
    'img_size': None,
}


def min_max_norm(
    img: np.ndarray, max_val: int = None, mask: np.ndarray = None,
    percentiles: tuple = None, dtype: str = None
) -> np.ndarray:
    """
    Scales images to be in range [0, 2**bits]

    Args:
        img (np.ndarray): Image to be scaled.
        max_val (int, optional): Value to scale images
            to after normalization. Defaults to None.
        mask (np.ndarray, optional): Mask to use in the normalization process.
            Defaults to None which means no mask is used.
        percentiles (tuple, optional): Tuple of percentiles to obtain min and max
            values respectively. Defaults to None which means min and max are used.
        dtype (str, optional): Output datatype

    Returns:
        np.ndarray: Scaled image with values from [0, max_val]
    """
    if mask is None:
        mask = np.ones_like(img)

    # Find min and max among the selected voxels
    if percentiles is not None:
        perc = np.percentile(img[mask != 0], list(percentiles))
        img_max = perc[1]
        img_min = perc[0]
    else:
        img_max = np.max(img[mask != 0])
        img_min = np.min(img[mask != 0])

    if max_val is None:
        # Determine possible max value according to data type
        max_val = np.iinfo(img.dtype).max

    # Normalize
    img = ((img - img_min) / (img_max - img_min)) * max_val
    img = np.clip(img, 0, max_val)

    # Adjust data type
    img = img.astype(dtype) if dtype is not None else img
    return img


class Preprocessor():
    def __init__(
        self,
        hist_match_cfg: dict = DEFAULT_HIST_MATCH_CONFIG,
        normalization_cfg: dict = DEFAULT_NORM_CONFIG,
        resize_cfg: dict = DEFAULT_RESIZE_CONFIG,
        multi_atlas: np.ndarray = None,
        misa_atlas: np.ndarray = None,
        tissue_models: np.ndarray = None
    ) -> None:
        self.hist_match_cfg = hist_match_cfg
        self.normalization_cfg = normalization_cfg
        self.resize_cfg = resize_cfg
        self.multi_atlas = multi_atlas
        self.misa_atlas = misa_atlas
        self.tissue_models = tissue_models

    def preprocess(
        self, img: np.ndarray, brain_mask: np.ndarray = None, metadata: dict = None
    ) -> np.ndarray:
        if self.hist_match_cfg is not None:
            ref_img_path = self.hist_match_cfg['ref_img_path']
            ref_img_path = Path(ref_img_path)
            img = self.match_hist(img, brain_mask, ref_img_path)

        # Normalize image intensities
        if self.normalization_cfg is not None:
            normalization_cfg = self.normalization_cfg.copy()
            normalization_cfg['mask'] = brain_mask if brain_mask is not None else None
            if normalization_cfg['type'] == 'min_max':
                del normalization_cfg['type']
                img = min_max_norm(img, **normalization_cfg)
            else:
                raise Exception(
                    f'Normalization {self.normalization_cfg["type"]} not implemented'
                )

        # Resize img and registered atlas
        r_multi_atlas = None
        r_misa_atlas = None
        if self.resize_cfg is not None:
            img, metadata = self.resize(img, **self.resize_cfg, metadata=metadata)
            if self.multi_atlas is not None:
                r_multi_atlas, _ = self.resize(self.mni_atlas, **self.resize_cfg, metadata=metadata)
            if self.misa_atlas is not None:
                r_misa_atlas, _ = self.resize(self.misa_atlas, **self.resize_cfg, metadata=metadata)

        # Get tissue models labels for the image
        tm_labels = None
        if self.tissue_models is not None:
            tm_labels = self.get_tissue_models_labels(img, brain_mask)

        return img, r_multi_atlas, r_misa_atlas, tm_labels, metadata

    def resize(
        self, img: np.ndarray, voxel_size: tuple, interpolation_order: int, metadata: dict
    ) -> np.ndarray:

        ratio = metadata['spacing']/np.array(voxel_size)
        img = zoom(img, ratio, order=interpolation_order)
        metadata['spacing'] = voxel_size
        return img, metadata['spacing']

    def get_tissue_models_labels(self, img: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
        t1_vector = img[brain_mask != 0].flatten()
        n_classes = self.tissue_models.shape[0]
        prob_map = np.zeros((n_classes, len(t1_vector)))
        t1_vector[t1_vector == 255] = 254
        for c in range(n_classes):
            prob_map[c, :] = self.tissue_models[c, t1_vector]
        prob_map = np.argmax(prob_map, axis=0)
        tm_labels_vol = brain_mask.flatten()
        tm_labels_vol[tm_labels_vol != 0] = prob_map + 1
        tm_labels_vol = tm_labels_vol.reshape(img.shape)
        return tm_labels_vol

    def match_hist(self, img: np.ndarray, brain_mask: np.ndarray, ref_img_path: Path) -> np.ndarray:
        ref_img = sitk.GetArrayFromImage(sitk.ReadImage(str(ref_img_path)))
        ref_img_brain_mask_path = ref_img_path.parent / ref_img_path.name.replace('n4', 'brain_mask')
        ref_img_brain_mask = sitk.GetArrayFromImage(sitk.ReadImage(str(ref_img_brain_mask_path)))
        img[brain_mask != 0] = match_histograms(
            img[brain_mask != 0], ref_img[ref_img_brain_mask != 0])
        return img
