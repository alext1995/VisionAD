'''
Copyright © 2024 Alexander Taylor
'''
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
import numpy as np
import torch
import traceback
from skimage import measure
from statistics import mean
from sklearn.metrics import auc 
import scipy.spatial.distance as dist
from skimage import measure
from statistics import mean
from sklearn.metrics import auc 
import pandas as pd
from functools import partial
import proportion_localised as pl

def produce_binary_metrics(labels, values):
    '''
    Produce the binary metrics for a given set of labels and values
    '''
    fpr, tpr, thresholds = roc_curve(labels, 
                                     values
                                     )

    precision, recall, thresholds = precision_recall_curve(labels, 
                                                           values
                                                           )

    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    threshold = thresholds[np.argmax(F1_scores)]
    predictions = (values >= threshold).astype(int)
    fpr_optim = np.mean(predictions > labels)
    fnr_optim = np.mean(predictions < labels)
    precision_optim = precision[np.argmax(F1_scores)]
    recall_optim    = recall[np.argmax(F1_scores)]

    return {"threshold": float(threshold), 
            "fpr_optim": float(fpr_optim), 
            "fnr_optim": float(fnr_optim), 
            "precision_optim": float(precision_optim),
            "recall_optim": float(recall_optim),
            "F1": float(np.max(F1_scores)),
            "precisions": [float(item) for item in precision[::len(precision)//200+1]],
            "recalls": [float(item) for item in recall[::len(recall)//200+1]],
            "fprs": [float(item) for item in fpr[::len(fpr)//200+1]],
            "tprs": [float(item) for item in tpr[::len(tpr)//200+1]]}


def imagewise_AUC(heatmap_set, image_score_set, targets_set, paths_set):
    '''
    Calculate the imagewise AUC for the given data
    '''
    out = {}
    regular = image_score_set["image_score_set_regular"]
    novel = image_score_set["image_score_set_novel"]

    further_info = {}
    for key in regular.keys():
        reg = regular[key]
        nov = novel[key]

        labels = np.concatenate((np.ones(nov.shape[0]), np.zeros(reg.shape[0])))
        values = np.concatenate((nov, reg))
        try:
            score = roc_auc_score(labels, values)
            if score>0.5:
                out[key] = score
                flip = 1
            else:
                out[key] = 1-score
                flip = -1

            further_info[key] = produce_binary_metrics(labels, flip*values)
        except:
            print(f"Unable to calculate imagewise_AUC for key {key}")
    return out,further_info

def pixelwise_AUC(heatmap_set, image_score_set, targets_set, paths_set, novel_only=False):
    '''
    Calculate the pixelwise AUC for the given data
    '''
    out = {}
    novel_predictions = heatmap_set["heatmap_set_novel"]
    regular_predictions = heatmap_set["heatmap_set_regular"]
    further_info = {}

    for key in novel_predictions.keys():
        if novel_only:
            pred = np.array(novel_predictions[key].ravel())
            true = np.array(targets_set["targets_novel"].ravel().int())
        else:
            regular_preds = np.array(regular_predictions[key].ravel())
            true = np.concatenate((np.array(targets_set["targets_novel"].ravel().int()),
                                  np.zeros(len(regular_preds))))
            pred = np.concatenate((np.array(novel_predictions[key].ravel()),
                                  regular_preds))
            
        try:
            score = roc_auc_score(true, 
                                  pred)
            if score>0.5:
                out[key] = score
                flip = 1
            else:
                out[key] = 1-score
                flip = -1

            further_info[key] = produce_binary_metrics(np.array(targets_set["targets_novel"].ravel().int()), 
                                                       flip*np.array(novel_predictions[key].ravel()))
            further_info[key] = {}
        except:
            print(f"Unable to calculate pixelwise_AUC for key {key}")
            print(traceback.format_exc())
        
    return out,further_info

pd.options.mode.chained_assignment = None
def compute_pro(masks: np.ndarray, amaps: np.ndarray, num_th: int = 200) -> None:
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    Thanks to https://github.com/hq-deng/RD4AD/blob/main/test.py for this great implementation of AUPRO,
    amendments have been made to allow it to handle reversed data, i.e. better detection at as threshold decreases
    and to make it much faster
    """
    masks = masks[:,0]
    amaps = amaps[:,0]

    if not isinstance(masks, np.ndarray):
        masks = masks.numpy()
    if not isinstance(amaps, np.ndarray):
        amaps = amaps.numpy()

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    infos = []
    aupros = []
    threshold_range = np.arange(min_th, max_th, delta)

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    axes_ids = []
    for binary_amap, mask in zip(binary_amaps, masks):
        a_axes_ids = []
        for region in measure.regionprops(measure.label(mask)):
            axes0_ids = region.coords[:, 0]
            axes1_ids = region.coords[:, 1]
            a_axes_ids.append((region.area, axes0_ids, axes1_ids))
        axes_ids.append(a_axes_ids)
        

    inverse_masks = 1 - masks
    inverse_masks_sum = inverse_masks.sum()
    for th in threshold_range[1:]:
        cond = amaps >= th
        binary_amaps[cond] = 1
        binary_amaps[~cond] = 0

        pros = []
        for binary_amap, mask, a_axes_ids in zip(binary_amaps, masks, axes_ids):
            for item in a_axes_ids:
                area, axes0_ids_, axes1_ids_ = item
                tp_pixels = binary_amap[axes0_ids_, axes1_ids_].sum()
                pros.append(tp_pixels / area)

        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks_sum

        df = pd.concat((df, pd.DataFrame({"pro": mean(pros), "pro_rev": 1-mean(pros), "fpr": fpr, "fpr_rev": 1-fpr, "threshold": th}, index=[0])), ignore_index=True)


    for reverse in [False, True]:
        try:
            if not reverse:
                df_normalised = df[df["fpr"] < 0.3]
                df_normalised["fpr"] = df_normalised["fpr"] / df_normalised["fpr"].max()

                pro_auc = auc(df_normalised["fpr"], df_normalised["pro"])

                infos.append({"fpr_crop_normalised": [item for item in df_normalised["fpr"]],
                            "pro_crop_normalised": [item for item in df_normalised["pro"]],
                            "fpr": [item for item in df["fpr"]],
                            "pro": [item for item in df["pro"]],
                            })
                aupros.append(pro_auc)
            else:
                df_normalised = df[df["fpr_rev"] < 0.3]
                df_normalised["fpr_rev"] = df_normalised["fpr_rev"] / df_normalised["fpr_rev"].max()

                pro_auc = auc(df_normalised["fpr_rev"], df_normalised["pro_rev"])

                infos.append({"fpr_crop_normalised": [item for item in df_normalised["fpr_rev"]],
                            "pro_crop_normalised": [item for item in df_normalised["pro_rev"]],
                            "fpr": [item for item in df["fpr_rev"]],
                            "pro": [item for item in df["pro_rev"]],
                            })
                aupros.append(pro_auc)
        except:
            aupros.append(0)
            infos.append({})

    return max(aupros), infos[np.argmax(aupros)]
    
def pixelwise_AUPRO(heatmap_set, image_score_set, targets_set, paths_set):
    '''
    Function to calculate the pixelwise AUPRO
    '''
    out = {}
    further_info = {}
    novel = heatmap_set["heatmap_set_novel"]
    for key in novel.keys():
        try:
            score, info = compute_pro(np.array(targets_set["targets_novel"].int()), 
                                np.array(novel[key]))
            out[key] = score
            further_info[key] = info
        except:
            print(f"Unable to calculate pixelwise_AUPRO for key {key}")
            print(traceback.format_exc())
    return out,further_info

class PLWrapper:
    '''
    Wrapper for the proportion localised metric. This takes the data in the format given
    by VisionAD, and calls the calculate function from the proportion_localised package.
    See
    https://pypi.org/project/proportion-localised
    https://github.com/alext1995/proportion_localised
    for more information on this metric
    '''
    def __init__(self):
        self.dictionary = {}

    def __call__(self, heatmap_set, image_score_set, targets_set, paths_set, 
                        IoU_limit, min_target_scale=1/8, n_thresholds=25,
                         overlap_limit=1/3):
        targets = targets_set["targets_novel"]
        out = {}
        further_info = {}
        for key, heatmaps in heatmap_set['heatmap_set_novel'].items():
            dict_key = f"{key}_{min_target_scale}_{n_thresholds}_{overlap_limit}"
            
            if dict_key in self.dictionary:
                score, data_out = self.dictionary[dict_key]
            else:
                score, data_out = pl.calculate(heatmaps[:,0], targets[:,0], IoU_limit,
                                                min_target_scale=min_target_scale,
                                                n_thresholds=n_thresholds,
                                                overlap_limit=overlap_limit)
                self.dictionary[dict_key] = (score, data_out)
                
            out[key] = score
            further_info[key] = data_out
          
        return out, further_info
            
    def reset(self):
        self.dictionary = {}

pl_wrapper = PLWrapper()

## the items from the metric list are called by the wrapper code
metric_list = {"Imagewise_AUC": imagewise_AUC, 
               "Pixelwise_AUC": partial(pixelwise_AUC, novel_only=False),
               "Pixelwise_AUC_anom_only": partial(pixelwise_AUC, novel_only=True),
               "Pixelwise_AUPRO": pixelwise_AUPRO,
               "PL"   : partial(pl_wrapper, IoU_limit=0.3),
               }

## this allows certain metrics to be calculated asynchronous in the groups shown below 
metric_key_list_async  = [["Imagewise_AUC", "Pixelwise_AUC"],
                            ["Pixelwise_AUPRO"],
                            ["Pixelwise_AUC_anom_only",
                             "PL",],]

def reset_all():
    '''
    This removes any cached information from the metrics (only relevant for PL).
    This must be done every epoch to ensure that the metric is calculated correctly.
    '''
    pl_wrapper.reset()

