import logging
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
from utils.utils import save_img_from_array_using_referece

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Define necessary paths relative to this file
this_file_path = Path('__file__').resolve()
data_path = this_file_path.parent / 'data'

for partition in ['train', 'val']:
    img_path = data_path / f'{partition}_set'
    logging.info(f'Obtaining brain tissue masks from {partition} labels...')
    for img_filepath in tqdm(img_path.iterdir(), total=len(list(img_path.iterdir()))):
        # Load the labels image and get the binary array
        brain_mask = sitk.ReadImage(str(img_filepath / f'{img_filepath.name}_seg.nii.gz'))
        bm_array = sitk.GetArrayFromImage(brain_mask)
        bm_array = np.where(bm_array > 0, 255, 0).astype('uint8')

        # Store images
        out_path = img_filepath / f'{img_filepath.name}_brain_mask.nii.gz'
        save_img_from_array_using_referece(bm_array, brain_mask, out_path)

logging.info('Brain tissue masks obteined succesfully.')
