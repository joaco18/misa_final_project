import logging
import pickle

import numpy as np

from pathlib import Path
from tqdm import tqdm

from dataset import MisaFPDataset

logging.basicConfig(level=logging.INFO, format='%(message)s')


def main():
    # Define data_path relative to this file
    this_file_path = Path('__file__').resolve()
    data_path = this_file_path.parent / 'data'

    # Reference values
    likelihoods = np.zeros((4, 256))

    # Run model over the complete dataset
    percentiles = None
    logging.info('Defining dataset...')
    train_dataset = MisaFPDataset(
        datapath=data_path,
        tissue_models_filepath=None,
        modalities=['T1'],
        pathologies=['normal'],
        partitions=['train'],
        load_atlases=False,
        normalization_cfg={
            'type': 'min_max',
            'max_val': 255,
            'mask': None,
            'percentiles': percentiles,
            'dtype': 'uint8'
        },
        skull_stripping=False,
        resize_cfg=None,
    )

    logging.info('Computing tissue models...')
    for idx in tqdm(range(len(train_dataset))):
        case = train_dataset[idx]
        # Accumulate the counts for each intensity across images in train set
        for c in [0, 1, 2, 3]:
            likelihoods[c, :] += np.histogram(
                case['t1'][case['ground_truth'] == c], bins=256, range=[0, 256])[0]

    # Obtain pdf and keep only pdfs of tissues
    likelihoods = likelihoods / np.sum(likelihoods, axis=1)[:, None]
    likelihoods = likelihoods[1:, :]

    # Get tissue models
    tissue_sums = np.sum(likelihoods, axis=0)[None, :]
    tissue_models = likelihoods / tissue_sums
    if percentiles is None:
        tissue_models[:, 225:] = np.array([0, 1, 0])[:, None]

    # Store tissue models
    tm_path = data_path / 'tissue_models'
    tm_path.mkdir(exist_ok=True, parents=True)
    with open((tm_path / 'tissue_models_3C.pkl'), 'wb') as f:
        pickle.dump(tissue_models, f)
    logging.info('Tissue models computed succefully.')


if __name__ == '__main__':
    main()
