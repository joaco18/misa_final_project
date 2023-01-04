import numpy as np


def brain_tissue_segmentation_tm(
    t1: np.ndarray, brain_mask: np.ndarray, tissue_models: np.ndarray
):
    """Computes brain tissue segmentation in single approach
    Args:
        t1 (np.ndarray): T1 volume
        brain_mask (np.ndarray): bask of brain tissues (region to classify)
        tissue_models (np.ndarray, optional): tissue models to use for segmentation.
    """
    # Define data
    t1_vector = t1[brain_mask != 0].flatten()
    n_classes = tissue_models.shape[0]
    preds = np.zeros((n_classes, len(t1_vector)))
    t1_vector[t1_vector == 255] = 254
    for c in range(n_classes):
        preds[c, :] = tissue_models[c, t1_vector]
    preds = np.argmax(preds, axis=0)
    predictions = brain_mask.flatten()
    predictions[predictions != 0] = preds + 1
    t1_seg_res = predictions.reshape(t1.shape)
    return t1_seg_res


def brain_tissue_segmentation_prob_map(
    brain_mask: np.ndarray, tissue_prob_maps: np.ndarray
):
    """Computes brain tissue segmentation in single approach
    Args:
        t1 (np.ndarray): T1 volume
        brain_mask (np.ndarray): bask of brain tissues (region to classify)
        tissue_prob_maps (np.ndarray, optional): size -> [n_class, [volume_shape]]
    """
    # Define data
    pred = np.argmax(tissue_prob_maps, axis=0)
    t1_seg_res = np.where(brain_mask != 0, pred, 0)
    return t1_seg_res


def brain_tissue_segmentation_tm_prob_map(
    t1: np.ndarray, brain_mask: np.ndarray,
    tissue_models: np.ndarray, tissue_prob_maps: np.ndarray
):
    """Computes brain tissue segmentation in single approach
    Args:
        t1 (np.ndarray): T1 volume
        brain_mask (np.ndarray): bask of brain tissues (region to classify)
        tissue_models (np.ndarray, optional): tissue models to use for segmentation.
        tissue_prob_maps (np.ndarray, optional): size -> [n_class, [volume_shape]]
    """
    # Define datas
    brain_mask = brain_mask.flatten()
    t1_vector = t1.flatten()[brain_mask != 0]
    prob_vects = tissue_prob_maps.reshape((tissue_prob_maps.shape[0], -1))
    prob_vects = prob_vects[1:, :]
    prob_vects = prob_vects[:, brain_mask != 0]

    n_classes = tissue_models.shape[0]
    preds = np.zeros((n_classes, len(t1_vector)))
    t1_vector[t1_vector == 255] = 254
    for c in range(n_classes):
        preds[c, :] = tissue_models[c, :][t1_vector]
    preds *= prob_vects
    preds = np.argmax(preds, axis=0)

    predictions = brain_mask.copy()
    predictions[brain_mask != 0] = preds + 1
    t1_seg_res = predictions.reshape(t1.shape)
    return t1_seg_res
