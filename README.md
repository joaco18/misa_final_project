# MISA - Final Project
## Kaouther Mouheb & Joaquin Oscar Seia



## 1. Repository content

&emsp;This repository contains the code for Medical Image Segmentation and Applications final project.

&emsp; The following steps and methods were done with this code:
- Brain mask extraction and N4 bias field correction of the volumes. This was done with the notebook notebooks/n4_bias_field_correction.ipynb
- Registration. We registered all volumes in train+validation set to all volumes in train+validation set. This was made with two objectives:
    - Determining the best choice of basic reference image in the train set to construct a generic probabilistic atlas that was later going to be registered to every case in the train+validation set. 
    - Perform multi-atlas segmentation.
    All the registerings were done using the notebook: notebooks/registrations.ipynb
- EM-based segementations. This were run using experiments/em_pred.py code and the configuration file in experiments/em_pred_config.yaml
- Result analysis of EM-based segmentations: done with notebooks/result_analysis_em.ipynb
- The tissue models, were obtained using the notebook: notebooks/tissue_models.ipynb
- Simple Segmenters. Which includes segementing based on tissue models and on probability atlases only. This were run using notebooks/simple_segmenters.ipynb notebook.
- Result analysis of Simple Segmenters' segmentations: done with notebooks/result_analysis_simple_seg.ipynb

- Following the mentioned notebooks you will find references to other pieces of code adecuately organized in other folders, like utils, preprocessing and postprocessing.

## 2. Instructions for contributers

&emsp; The presented pipeline can be fully reproduced locally. Below we provide  BASH commands, which can be run in an Unix/Unix-like OS (Mac OS, GNU-Linux) and CMD comands for windows user (¬¬ consider changing to GNU-Linux, your life will be better).

### 2.1 Setting up the environment

- Create the environment

    > Unix:
    ```bash
    conda create -n misa_fp python==3.9.13 anaconda -y &&
    conda activate misa_fp
    ```

    >Windows:
    ```bash
    conda create -n misa_fp python==3.9.13 anaconda -y && conda activate misa_fp
    ```

- Install requirements
    >Both OS:
    ```bash
    pip install -r requirements.txt
    ```

- Add current repository path to PYTHONPATH

    > Unix:
    ```bash
    export PYTHONPATH="${PYTHONPATH}:/path/to/your/project/"
    ```

    > Windows:
    ```bash
    set PYTHONPATH=%PYTHONPATH%;/path/to/your/project/
    ```

### 2.2 Run pipeline as developer
- 2.2.1 **Download the database**

    > Unix:
    ```bash
    mkdir -p data &&
    cd data/ &&
    gdown https://drive.google.com/uc?id=1H_nEwJybDDA0Z1UfYhpIZt5LMFc3QzU_ &&
    unzip data.zip &&
    rm data.zip &&
    cd ../
    ```
    > Windows:
    ```bash
    mkdir -p data && cd data/ && gdown https://drive.google.com/uc?id=1H_nEwJybDDA0Z1UfYhpIZt5LMFc3QzU_ && tar -xf data.zip && del data.zip && cd ../
    ```

    > Alternative:

    &emsp; If you don't want to download from command line. Here is the drive [link](https://drive.google.com/drive/folders/1H_nEwJybDDA0Z1UfYhpIZt5LMFc3QzU_?usp=share_link)

    &emsp; The directories should be misa_final_project/data/[content of data zip]



### 4.4 Recommendations to developers

- The code in Medvision is developed following:
    - numpy docstring format
    - flake8 linter
    - characters per line: 100