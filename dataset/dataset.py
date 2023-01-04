import pickle

import numpy as np
import pandas as pd
import SimpleITK as sitk

from pathlib import Path
from typing import Dict, List

from preprocessing.preprocessing import Preprocessor
from utils.utils import extract_metadata

this_file_path = Path().resolve()
data_path = this_file_path.parent / 'data'
tm_filepath = data_path / 'tissue_models' / 'tissue_models_3C.pkl'


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


class IBSRDataset():
    def __init__(
        self,
        datapath: Path = data_path,
        tissue_models_filepath: Path = tm_filepath,
        modalities: List[str] = ['T1'],
        pathologies: List[str] = ['normal'],
        partitions: List[str] = ['train', 'test'],
        case_selection: List[str] = ['all'],
        load_atlases: bool = True,
        hist_match_cfg: dict = DEFAULT_HIST_MATCH_CONFIG,
        normalization_cfg: dict = DEFAULT_NORM_CONFIG,
        resize_cfg: dict = DEFAULT_RESIZE_CONFIG,
    ) -> None:

        # Set the atributes
        self.datapath = datapath
        self.tissue_models_filepath = tissue_models_filepath
        self.modalities = modalities
        self.pathologies = pathologies
        self.partitions = partitions
        self.load_atlases = load_atlases
        self.case_selection = case_selection

        # Load the dataset csv
        self.df = pd.read_csv(self.datapath/'misa_fp_dataset.csv', index_col=0)

        # Load tissue models
        self.tissue_models = None
        if self.tissue_models_filepath is not None:
            with open((self.tissue_models_filepath), 'rb') as f:
                self.tissue_models = pickle.load(f)

        # Filter the desired cases
        self.filter_by_modality()
        self.filter_by_pathologies()
        self.filter_by_partitions()
        if 'all' not in self.case_selection:
            self.filter_by_case_selection()

        # Define the preprocessings
        self.t1_preprocessor = Preprocessor(
            hist_match_cfg=hist_match_cfg,
            normalization_cfg=normalization_cfg,
            resize_cfg=resize_cfg,
            multi_atlas=None,
            misa_atlas=None,
            tissue_models=self.tissue_models
        )
        if resize_cfg is not None:
            resize_cfg_mask = resize_cfg.copy()
            resize_cfg_mask['interpolation'] = 'NearestNeighbour'
        else:
            resize_cfg_mask = None

        self.label_preprocessor = Preprocessor(
            hist_match_cfg=None,
            normalization_cfg=None,
            resize_cfg=resize_cfg_mask,
            multi_atlas=None,
            misa_atlas=None,
            tissue_models=None
        )

        # Check if filtering left sth in the dataframe
        assert len(self.df) != 0, 'Dataset is empy, check your filtering parameters'

    def filter_by_modality(self):
        self.df = self.df.loc[self.df.modality.isin(self.modalities)]
        self.df.reset_index(drop=True, inplace=True)

    def filter_by_pathologies(self):
        self.df = self.df.loc[self.df.pathology.isin(self.pathologies)]
        self.df.reset_index(drop=True, inplace=True)

    def filter_by_partitions(self):
        self.df = self.df.loc[self.df.partition.isin(self.partitions)]
        self.df.reset_index(drop=True, inplace=True)

    def filter_by_case_selection(self):
        self.df = self.df.loc[self.df.case.isin([case for case in self.case_selection])]
        self.df.reset_index(drop=True, inplace=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> Dict:
        df_row = self.df.loc[idx, :].squeeze()
        img_name = df_row.case
        img_path = (self.datapath / df_row.img_path).parent

        sample = {}
        sample['id'] = img_name
        # Load image, labels and brain mask
        sample['t1'] = sitk.ReadImage(str(img_path / f'{img_name}_n4.nii.gz'), sitk.sitkUInt8)
        sample['ref_metadata'] = extract_metadata(sample['t1'])
        sample['t1'] = sitk.GetArrayFromImage(sample['t1'])

        sample['ground_truth'] = sitk.GetArrayFromImage(
            sitk.ReadImage(str(img_path / f'{img_name}_seg.nii.gz'))
        ).astype('int')

        sample['brain_mask'] = sitk.GetArrayFromImage(
            sitk.ReadImage(str(img_path / f'{img_name}_brain_mask.nii.gz'))
        )
        sample['brain_mask'] = np.where(sample['brain_mask'] != 0, 255, 0).astype('uint8')

        # Preprocess
        sample['t1'], _, _, sample['tissue_models_labels'], sample['ref_metadata'] = \
            self.t1_preprocessor.preprocess(sample['t1'], sample['brain_mask'], sample['ref_metadata'])
        sample['ground_truth'], _, _, _, _ = \
            self.label_preprocessor.preprocess(sample['ground_truth'], metadata=sample['ref_metadata'])
        sample['brain_mask'], _, _, _, _ = \
            self.label_preprocessor.preprocess(sample['brain_mask'], metadata=sample['ref_metadata'])

        # Load atlases
        if self.load_atlases:
            # sample['tpm_mni'] = sitk.GetArrayFromImage(sitk.ReadImage(
            #     str(img_path / f'{img_name}_mni_atlas.nii.gz')
            # ))
            # sample['tpm_mni'] = np.clip(sample['tpm_mni'], a_min=0, a_max=1)
            sample['tpm_misa'] = sitk.GetArrayFromImage(sitk.ReadImage(
                str(img_path / f'{img_name}_atlas.nii.gz')
            ))
            sample['tpm_misa'] = np.clip(sample['tpm_misa'], 0, 1)
            sample['tpm_multi'] = sitk.GetArrayFromImage(sitk.ReadImage(
                str(img_path / f'{img_name}_multi-atlas.nii.gz')
            ))
            sample['tpm_multi'] = np.clip(sample['tpm_multi'], 0, 1)
        return sample
