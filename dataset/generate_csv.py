import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Define data_path relative to this file
this_file_path = Path('__file__').resolve()
data_path = this_file_path.parent / 'data'

# Define holder for all csv data
df = []

# Generate train information
for partition in ['train', 'val', 'test']:
    logging.info(f'Reading {partition} data...')
    train_img_path = data_path / f'{partition}_set'
    for img_path in train_img_path.iterdir():
        row = [
            img_path.name,
            '/'.join(str(img_path).split('/')[-2:] + [f'{img_path.name}.nii.gz']),
            'normal',
            'T1',
            partition
        ]
        df.append(row)

# Store the csv file
df = pd.DataFrame(df, columns=['case', 'img_path', 'pathology', 'modality', 'partition'])
df.to_csv(data_path / 'misa_fp_dataset.csv')
logging.info('Csv file generated succesfully')
