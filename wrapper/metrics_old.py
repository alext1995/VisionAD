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

def produce_binary_metrics(labels, values):
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


def imagewise_AUC(heatmap_set, loss_set, targets_set, paths_set):
    
    out = {}
    regular = loss_set["loss_set_regular"]
    novel = loss_set["loss_set_novel"]

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

def pixelwise_AUC(heatmap_set, loss_set, targets_set, paths_set, novel_only=False):
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

## thanks to https://github.com/hq-deng/RD4AD/blob/main/test.py for this great implementation of PRO
## amendments have been made to allow it to handle reversed data, i.e. better detection at as threshold decreases
## and to make it much faster
pd.options.mode.chained_assignment = None
def compute_pro(masks: np.ndarray, amaps: np.ndarray, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
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
    
def pixelwise_AUPRO(heatmap_set, loss_set, targets_set, paths_set):
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

def get_region_center(region):
    return [(region.coords[:, 0].min()+region.coords[:, 0].max())/2, (region.coords[:, 1].min()+region.coords[:, 1].max())/2]

def calculate_IoU(predictions, mask):
    return (np.logical_and(predictions, mask)).sum()/(np.logical_or(predictions, mask)).sum()

def PL_family_old(heatmaps, targets, min_target_scale=8, n_thresholds=25):
    IoUs = []
    
    min_pred = heatmaps.min()
    max_pred = heatmaps.max()
    
    if not isinstance(heatmaps, np.ndarray):
        heatmaps = heatmaps.numpy()
    if not isinstance(targets, np.ndarray):
        targets = targets.numpy()
    
    thresholds = [threshold for threshold in np.linspace(min_pred, max_pred, n_thresholds+2)[1:-1]]

    IoUs = []
    for heatmap, target in zip(heatmaps, targets):

        # gather the prediction_masks 
        regions = [region for region in measure.regionprops(measure.label(target>0))]
        region_centers = [get_region_center(region) for region in regions]

        xx, yy = np.meshgrid(np.arange(256), np.arange(256))
        distances = dist.cdist(np.array([xx.ravel(), 
                                         yy.ravel()]).T, 
                               region_centers, 
                               metric='euclidean').reshape(256, 256, len(region_centers))

        prediction_masks = []
        temp_masks = []
        for region_ind, region in enumerate(regions):

            temp_mask = np.zeros((256, 256))
            temp_mask[tuple([[item[0] for item in region.coords], [item[1] for item in region.coords]])] = 1

            bottom = min(region.coords[:,0])
            top = max(region.coords[:,0])

            right = max(region.coords[:,1])
            left = min(region.coords[:,1])

            width = right - left
            height = top - bottom

            center = region_centers[region_ind]

            if width<256//min_target_scale:
                right = center[1]+256//(min_target_scale*2)
                left = center[1]-256//(min_target_scale*2)

            if height<256//min_target_scale:
                top = center[0]+256//(min_target_scale*2)
                bottom = center[0]-256//(min_target_scale*2)

            temp_mask[int(bottom):int(top), int(left):int(right)] = 256

            closest_pixels = np.argmin(distances.swapaxes(0, 1), axis=-1)==region_ind
            prediction_mask = np.zeros((256, 256))
            prediction_mask[closest_pixels] = 1
            prediction_mask[temp_mask>0] = 1
            prediction_masks.append(prediction_mask)
            temp_masks.append(temp_mask)

        # loop over the prediction_masks
        for prediction_mask, temp_mask in zip(prediction_masks, temp_masks):
            IoUs_per_heatmap_overthresholds = []
            for threshold in thresholds:
                nov_threshold = heatmap>threshold
                predictions = np.logical_and(nov_threshold, prediction_mask)
                IoU = calculate_IoU(predictions=predictions, mask=temp_mask)
                IoUs_per_heatmap_overthresholds.append(IoU)
            IoUs.append(IoUs_per_heatmap_overthresholds)
    return np.array(IoUs).T

class DataHolderPL_old:
    def __init__(self):
        self.dictionary = {}
        
    def __call__(self, heatmap_set, loss_set, targets_set, paths_set, 
                        IoU_limit, version, min_target_scale=8, n_thresholds=25):
        
        targets = targets_set["targets_novel"]
        out = {}
        further_info = {}
        for key, heatmaps in heatmap_set['heatmap_set_novel'].items():
            dict_key = f"{key}_{min_target_scale}_{n_thresholds}"
            if dict_key not in self.dictionary:
                array = PL_family_old(heatmaps[:,0], 
                                     targets[:,0],
                                     min_target_scale=min_target_scale, 
                                     n_thresholds=n_thresholds)
                self.dictionary[dict_key] = array
                
                array_rev = PL_family_old(-1*heatmaps[:,0], 
                                         targets[:,0],
                                         min_target_scale=min_target_scale, 
                                         n_thresholds=n_thresholds)
                self.dictionary[dict_key+"_rev"] = array_rev
            else:
                array = self.dictionary[dict_key]
                array_rev = self.dictionary[dict_key+"_rev"]
            if version=="st":
                score = (array>IoU_limit).mean(axis=1).max()
                score_rev = (array_rev>IoU_limit).mean(axis=1).max()
            else:
                score = (array.max(axis=0)>IoU_limit).mean()
                score_rev = (array_rev.max(axis=0)>IoU_limit).mean()
            best = np.argmax([score, score_rev])   
            out[key] = float([score, score_rev][best]) 
            further_info[key] = [list(row) for row in [array, array_rev][best]]
        
        return out, further_info
            
    def reset(self):
        self.dictionary = {}

### new PL code

from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
import cv2

def get_rotation_info(trial_mask):
    contours, _ = cv2.findContours(trial_mask.astype(np.uint8)*255, 
                                   cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)

    try:                
        all_points = np.concatenate([contour[:,0,:] for contour in contours])
        hull = cv2.convexHull(all_points)
    except:
        hull = contours[np.argmax([len(item) for item in contours])]
        
    ((center_x, center_y), (width, height), angle_of_rotation) = cv2.minAreaRect(hull)
    return ((center_x, center_y), (width, height), angle_of_rotation)

def create_bounding_box_mask(mask_shape, bounding_box_info):
    box = cv2.boxPoints(bounding_box_info)
    box = np.int0(box)
    mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.drawContours(mask, [box], 0, 255, -1)
    return mask

def check_overlap_rotated(box1, box2):
    # Create polygons representing rotated bounding boxes
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)
    
    # Calculate intersection area
    intersection_area = poly1.intersection(poly2).area
    
    # Calculate area of the first bounding box
    box1_area = poly1.area
    
    # Check if intersection area is at least 50% of the first bounding box area
    return (intersection_area/box1_area)

## get the min distance between the centers - cdist
def get_min_distance_indices(coordinates):
    distances = cdist(coordinates, coordinates)

    np.fill_diagonal(distances, np.inf)

    # Find the indices of the closest coordinates
    closest_indices = np.argmin(distances, axis=1)
    min_index = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
    min_index
    
    out = []
    for a, b in zip(*np.unravel_index(np.argsort(distances, axis=None), distances.shape)):
        if a!=b:
            out.append((a, b))

    return np.array(out[::2])

def build_graph(pairs):
    graph = {}
    for pair in pairs:
        a, b = pair
        if a not in graph:
            graph[a] = []
        if b not in graph:
            graph[b] = []
        graph[a].append(b)
        graph[b].append(a)
    return graph

def dfs(graph, start, visited, component):
    visited.add(start)
    component.append(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited, component)

def connected_components(pairs):
    graph = build_graph(pairs)
    visited = set()
    components = []
    for vertex in graph:
        if vertex not in visited:
            component = []
            dfs(graph, vertex, visited, component)
            components.append(component)
    return components

def sort_linked_sets(pairs):
    components = connected_components(pairs)
    return [sorted(component) for component in components]

def filter_bounding_boxes(bbs, overlap_limit, shape=(256, 256)):
    restart = True
    shape = (256, 256)
    
    bounding_box_infos = [item for item in bbs]
    
    merge_list = [] 
    
    coordinates = [item[0] for item in bounding_box_infos]
    dims = [item[1] for item in bounding_box_infos]
    out = get_min_distance_indices(coordinates)

#     mask1 = np.zeros((256, 256))
#     for item in bounding_box_infos:
    #     item = ((192.00003051757812, 154.5), (32, 71.5541763305664), 26.56505012512207)
    #         mask1 += (create_bounding_box_mask((256, 256), item)*0.2).astype(int)
#         drawcontour(mask1, item)
#     plt.figure()
#     plt.imshow(mask1)

    for indices in out:
        box1, box2 = coordinates[indices[0]], coordinates[indices[1]]
        box1area = sum(dims[indices[0]])
        box2area = sum(dims[indices[1]])

        bounding_box_info1 = cv2.boxPoints(bounding_box_infos[indices[0]])
        bounding_box_info2 = cv2.boxPoints(bounding_box_infos[indices[1]])

        if box1area<box2area:
            out = np.zeros(shape)
            out += create_bounding_box_mask(shape, bounding_box_infos[indices[0]])/5
            out += create_bounding_box_mask(shape, bounding_box_infos[indices[1]])/2
            out[out>0] = 255

            overlap = check_overlap_rotated(bounding_box_info1, bounding_box_info2)

        else:
            out = np.zeros(shape)
            out += create_bounding_box_mask(shape, bounding_box_infos[indices[0]])/5
            out += create_bounding_box_mask(shape, bounding_box_infos[indices[1]])/2

            overlap = check_overlap_rotated(bounding_box_info2, bounding_box_info1)

        if overlap>overlap_limit:
            out = np.zeros(shape)
            out += create_bounding_box_mask(shape, bounding_box_infos[indices[0]])/5
            out += create_bounding_box_mask(shape, bounding_box_infos[indices[1]])/2

            new_box = get_rotation_info(out)

            new_bounding_box_infos = [item for i, item in enumerate(bounding_box_infos) if i not in [indices[0], indices[1]]]
            merge_list.append([indices[0], indices[1]])

            #restart = True
            #break
      
    groups = sort_linked_sets(merge_list)
#         print(groups)
    boxes_out = []
    for group in groups:
        out = np.zeros(shape)
        for indice in group:
            out += create_bounding_box_mask(shape, bounding_box_infos[indice])
#             plt.figure()
#             plt.imshow(out)

        new_box = get_rotation_info(out)
        boxes_out.append(new_box)

    for indice in range(len(coordinates)):
        if not np.any([indice in group for group in groups]):
            boxes_out.append(bounding_box_infos[indice])
    return boxes_out

def make_scaled_bb(temp_mask, min_target_scale):
    ((center_x, center_y), (width, height), angle_of_rotation) = get_rotation_info(temp_mask)
    
    if width<256//min_target_scale:
        width = 256//min_target_scale

    if height<256//min_target_scale:
        height = 256//min_target_scale
        
    return ((center_x, center_y), (width, height), angle_of_rotation)

def get_overlapping_masks(bbs, overlap_limit=0.5):
    shape = (256, 256)
    bounding_box_infos = [item for item in bbs]
    
    coordinates = [item[0] for item in bounding_box_infos]
    dims = [item[1] for item in bounding_box_infos]
    out = get_min_distance_indices(coordinates)
    merge_list = []
    for indices in out:
        box1, box2 = coordinates[indices[0]], coordinates[indices[1]]
        box1area = sum(dims[indices[0]])
        box2area = sum(dims[indices[1]])

        bounding_box_info1 = cv2.boxPoints(bounding_box_infos[indices[0]])
        bounding_box_info2 = cv2.boxPoints(bounding_box_infos[indices[1]])

        if box1area<box2area:
            out = np.zeros(shape)
            out += create_bounding_box_mask(shape, bounding_box_infos[indices[0]])/5
            out += create_bounding_box_mask(shape, bounding_box_infos[indices[1]])/2
            out[out>0] = 255

            overlap = check_overlap_rotated(bounding_box_info1, bounding_box_info2)

        else:
            out = np.zeros(shape)
            out += create_bounding_box_mask(shape, bounding_box_infos[indices[0]])/5
            out += create_bounding_box_mask(shape, bounding_box_infos[indices[1]])/2

            overlap = check_overlap_rotated(bounding_box_info2, bounding_box_info1)
        
        if overlap>overlap_limit:
            out = np.zeros(shape)
            out += create_bounding_box_mask(shape, bounding_box_infos[indices[0]])/5
            out += create_bounding_box_mask(shape, bounding_box_infos[indices[1]])/2

            new_box = get_rotation_info(out)

            new_bounding_box_infos = [item for i, item in enumerate(bounding_box_infos) if i not in [indices[0], indices[1]]]
            merge_list.append([indices[0], indices[1]])

            #restart = True
            #break
      
    groups = sort_linked_sets(merge_list)
    
    # print(groups)
    for indice in range(len(coordinates)):
        if not np.any([indice in group for group in groups]):
            groups.append([indice])
    return groups

def merge_overlapping(individual_masks, scaled_bbs, min_target_scale, overlap_limit=0.5):
    groups = get_overlapping_masks(scaled_bbs, overlap_limit=overlap_limit)
    ## merge the groups here
    out_masks = []
    for group in groups:
        out_mask = np.zeros((256, 256))
        for ind in group:
            out_mask += individual_masks[ind]
        out_masks.append(out_mask)
        
    scaled_bbs_2 = [make_scaled_bb(item, min_target_scale) for item in out_masks]
    
    groups = get_overlapping_masks(scaled_bbs_2, overlap_limit=overlap_limit)
    out_masks_2 = []
    for group in groups:
        out_mask = np.zeros((256, 256))
        for ind in group:
            out_mask += out_masks[ind]
        out_masks_2.append(out_mask)
    scaled_bbs_3 = [make_scaled_bb(item, min_target_scale) for item in out_masks_2]
    
    return scaled_bbs_3

def target_to_rotated_scaled_merged_BBs(target, overlap_limit, min_target_scale, shape=(256, 256)):
    regions = [region for region in measure.regionprops(measure.label(target>0))]
    
    if len(regions)==1:
        return [make_scaled_bb(target, min_target_scale)]
    scaled_bbs = []
    individual_masks = []
    for region_ind, region in enumerate(regions):
        temp_mask = np.zeros((256, 256))
        temp_mask[tuple([[item[0] for item in region.coords], [item[1] for item in region.coords]])] = 1
        individual_masks.append(temp_mask)
        scaled_bbs.append(make_scaled_bb(temp_mask, min_target_scale))
    
    return merge_overlapping(individual_masks, scaled_bbs, min_target_scale, overlap_limit=overlap_limit)

def PL_family_rotation(heatmaps, targets, min_target_scale=8, n_thresholds=25, 
                       rotation=True, 
                       merge_close_anomalies=True,
                       overlap_limit=1/3,
                       rotated_scaled_merged=True):
    IoUs = []
    
    min_pred = heatmaps.min()
    max_pred = heatmaps.max()
    
    if not isinstance(heatmaps, np.ndarray):
        heatmaps = heatmaps.numpy()
    if not isinstance(targets, np.ndarray):
        targets = targets.numpy()
    
    thresholds = [threshold for threshold in np.linspace(min_pred, max_pred, n_thresholds+2)[1:-1]]
    
    all_prediction_masks = []
    all_temp_masks = []
    
    IoUs = []
    for heatmap, target in zip(heatmaps, targets):
        
        if rotated_scaled_merged and not rotation and not merge_close_anomalies:
            bounding_box_infos = target_to_rotated_scaled_merged_BBs(target, overlap_limit, min_target_scale, shape=(256, 256))
            xx, yy = np.meshgrid(np.arange(256), np.arange(256))
            
            prediction_masks = []
            temp_masks = []
        else:
            # gather the prediction_masks 
            regions = [region for region in measure.regionprops(measure.label(target>0))]
            region_centers = [get_region_center(region) for region in regions]

            xx, yy = np.meshgrid(np.arange(256), np.arange(256))
            distances = dist.cdist(np.array([xx.ravel(), 
                                             yy.ravel()]).T, 
                                   region_centers, 
                                   metric='euclidean').reshape(256, 256, len(region_centers))

            prediction_masks = []
            temp_masks = []
            bounding_box_infos = []
            if len(regions)==3:
                print_ = True
            else:
                print_ = False
            for region_ind, region in enumerate(regions):

                temp_mask = np.zeros((256, 256))
                temp_mask[tuple([[item[0] for item in region.coords], [item[1] for item in region.coords]])] = 1
                temp_mask_rotation = temp_mask.copy()

                if rotation:
                    ((center_x, center_y), (width, height), angle_of_rotation) = get_rotation_info(temp_mask_rotation)
                else:
                    bottom = min(region.coords[:,0])
                    top = max(region.coords[:,0])

                    right = max(region.coords[:,1])
                    left = min(region.coords[:,1])

                    width = right - left
                    height = top - bottom
                    center = region_centers[region_ind]
                    ((center_x, center_y), (width, height), angle_of_rotation) = ((float(center[1]), float(center[0])), 
                                                                                    (float(width), float(height),), 
                                                                                    0)

                if width<256//min_target_scale:
                    width = 256//min_target_scale

                if height<256//min_target_scale:
                    height = 256//min_target_scale

                bounding_box_infos.append(((center_x, center_y), (width, height), angle_of_rotation))

            if len(bounding_box_infos)>1 and merge_close_anomalies:
                bounding_box_infos = filter_bounding_boxes(bounding_box_infos, overlap_limit=overlap_limit, shape=(256, 256))
        
        region_centers_ = []
        for bounding_box_info in bounding_box_infos:
            center_x, center_y = bounding_box_info[0]
            region_centers_.append((center_y, center_x))
            
        # if not rotation:
        #     region_centers_ = region_centers
            
        distances = dist.cdist(np.array([xx.ravel(), 
                                         yy.ravel()]).T, 
                               region_centers_, 
                               metric='euclidean').reshape(256, 256, len(region_centers_))
                
        for region_ind, bounding_box_info in enumerate(bounding_box_infos):
            
            bb_mask = create_bounding_box_mask((256, 256), 
                                               bounding_box_info)
            
            closest_pixels = np.argmin(distances.swapaxes(0, 1), axis=-1)==region_ind
            prediction_mask = np.zeros((256, 256))
            prediction_mask[closest_pixels] = 1
            prediction_mask[bb_mask>0] = 1


            prediction_masks.append(prediction_mask)
            temp_masks.append(bb_mask)
        
        all_prediction_masks.append(prediction_masks)
        all_temp_masks.append(temp_masks)
        # loop over the prediction_masks
        for prediction_mask, temp_mask in zip(prediction_masks, temp_masks):
            IoUs_per_heatmap_overthresholds = []
            for threshold in thresholds:
                nov_threshold = heatmap>threshold
                predictions = np.logical_and(nov_threshold, prediction_mask)
                IoU = calculate_IoU(predictions=predictions, mask=temp_mask)
                IoUs_per_heatmap_overthresholds.append(IoU)
            IoUs.append(IoUs_per_heatmap_overthresholds)
    return np.array(IoUs).T, ((all_prediction_masks, all_temp_masks))

class DataHolderPL:
    def __init__(self):
        self.dictionary = {}
        
    def __call__(self, heatmap_set, loss_set, targets_set, paths_set, 
                        IoU_limit, version, min_target_scale=8, n_thresholds=25,
                         mask_rotation=True, merge_close_anomalies=True, overlap_limit=1/3):
        
        targets = targets_set["targets_novel"]
        out = {}
        further_info = {}
        for key, heatmaps in heatmap_set['heatmap_set_novel'].items():
            dict_key = f"{key}_{min_target_scale}_{n_thresholds}_rot{int(mask_rotation)}_mergecloseanom{int(merge_close_anomalies)}"
            if dict_key not in self.dictionary:
                array, _ = PL_family_rotation(heatmaps[:,0], 
                                              targets[:,0],
                                              min_target_scale=min_target_scale, 
                                              n_thresholds=n_thresholds,
                                              rotation=mask_rotation,
                                              merge_close_anomalies=merge_close_anomalies,
                                              overlap_limit=overlap_limit,
                                             )
                self.dictionary[dict_key] = array

                array_rev, _ = PL_family_rotation(-1*heatmaps[:,0], 
                                                  targets[:,0],
                                                  min_target_scale=min_target_scale, 
                                                  n_thresholds=n_thresholds,
                                                  rotation=mask_rotation,
                                                  merge_close_anomalies=merge_close_anomalies,
                                                  overlap_limit=overlap_limit)
                self.dictionary[dict_key+"_rev"] = array_rev
            else:
                array = self.dictionary[dict_key]
                array_rev = self.dictionary[dict_key+"_rev"]
            if version=="st":
                score = (array>IoU_limit).mean(axis=1).max()
                score_rev = (array_rev>IoU_limit).mean(axis=1).max()
            else:
                score = (array.max(axis=0)>IoU_limit).mean()
                score_rev = (array_rev.max(axis=0)>IoU_limit).mean()
            best = np.argmax([score, score_rev])   
            out[key] = float([score, score_rev][best]) 
            further_info[key] = [list(row) for row in [array, array_rev][best]]
        
        return out, further_info
            
    def reset(self):
        self.dictionary = {}


def PL_family_rotation_fixed(heatmaps, targets, min_target_scale=8, n_thresholds=25, 
                       rotation=True, 
                       merge_close_anomalies=True,
                       overlap_limit=1/3,
                       ):
    IoUs = []
    
    min_pred = heatmaps.min()
    max_pred = heatmaps.max()
    
    if not isinstance(heatmaps, np.ndarray):
        heatmaps = heatmaps.numpy()
    if not isinstance(targets, np.ndarray):
        targets = targets.numpy()
    
    thresholds = [threshold for threshold in np.linspace(min_pred, max_pred, n_thresholds+2)[1:-1]]
    
    all_prediction_masks = []
    all_temp_masks = []
    all_bounding_box_infos = []

    IoUs = []
    for heatmap, target in zip(heatmaps, targets):
        
       
        bounding_box_infos = target_to_rotated_scaled_merged_BBs(target, overlap_limit, min_target_scale, shape=(256, 256))
        xx, yy = np.meshgrid(np.arange(256), np.arange(256))
        
        prediction_masks = []
        temp_masks = []

        # else:
        #     print("Doing the second one")
        #     # gather the prediction_masks 
        #     regions = [region for region in measure.regionprops(measure.label(target>0))]
        #     region_centers = [get_region_center(region) for region in regions]

        #     xx, yy = np.meshgrid(np.arange(256), np.arange(256))
        #     distances = dist.cdist(np.array([xx.ravel(), 
        #                                      yy.ravel()]).T, 
        #                            region_centers, 
        #                            metric='euclidean').reshape(256, 256, len(region_centers))

        #     prediction_masks = []
        #     temp_masks = []
        #     bounding_box_infos = []
        #     if len(regions)==3:
        #         print_ = True
        #     else:
        #         print_ = False

        #     for region_ind, region in enumerate(regions):

        #         temp_mask = np.zeros((256, 256))
        #         temp_mask[tuple([[item[0] for item in region.coords], [item[1] for item in region.coords]])] = 1
        #         temp_mask_rotation = temp_mask.copy()

        #         if rotation:
        #             ((center_x, center_y), (width, height), angle_of_rotation) = get_rotation_info(temp_mask_rotation)
        #         else:
        #             bottom = min(region.coords[:,0])
        #             top = max(region.coords[:,0])

        #             right = max(region.coords[:,1])
        #             left = min(region.coords[:,1])

        #             width = right - left
        #             height = top - bottom
        #             center = region_centers[region_ind]
        #             ((center_x, center_y), (width, height), angle_of_rotation) = ((float(center[1]), float(center[0])), 
        #                                                                             (float(width), float(height),), 
        #                                                                             0)

        #         if width<256//min_target_scale:
        #             width = 256//min_target_scale

        #         if height<256//min_target_scale:
        #             height = 256//min_target_scale

        #         bounding_box_infos.append(((center_x, center_y), (width, height), angle_of_rotation))

        #     if len(bounding_box_infos)>1 and merge_close_anomalies:
        #         bounding_box_infos = filter_bounding_boxes(bounding_box_infos, overlap_limit=overlap_limit, shape=(256, 256))
            
        all_bounding_box_infos.append(bounding_box_infos)
        
        region_centers_ = []
        for bounding_box_info in bounding_box_infos:
            center_x, center_y = bounding_box_info[0]
            region_centers_.append((center_y, center_x))
            
        # if not rotation:
        #     region_centers_ = region_centers
            
        distances = dist.cdist(np.array([xx.ravel(), 
                                         yy.ravel()]).T, 
                               region_centers_, 
                               metric='euclidean').reshape(256, 256, len(region_centers_))
                
        for region_ind, bounding_box_info in enumerate(bounding_box_infos):
            
            bb_mask = create_bounding_box_mask((256, 256), 
                                               bounding_box_info)
            
            closest_pixels = np.argmin(distances.swapaxes(0, 1), axis=-1)==region_ind
            prediction_mask = np.zeros((256, 256))
            prediction_mask[closest_pixels] = 1
            prediction_mask[bb_mask>0] = 1


            prediction_masks.append(prediction_mask)
            temp_masks.append(bb_mask)
        
        all_prediction_masks.append(prediction_masks)
        all_temp_masks.append(temp_masks)

        # loop over the prediction_masks
        for prediction_mask, temp_mask in zip(prediction_masks, temp_masks):
            IoUs_per_heatmap_overthresholds = []
            for threshold in thresholds:
                nov_threshold = heatmap>threshold
                predictions = np.logical_and(nov_threshold, prediction_mask)
                IoU = calculate_IoU(predictions=predictions, mask=temp_mask)
                IoUs_per_heatmap_overthresholds.append(IoU)
            IoUs.append(IoUs_per_heatmap_overthresholds)

    array_out = np.array(IoUs).T
    array_out_ = (array_out>0.3).mean(axis=1)
    score = array_out_.max()
    thres_ind = array_out_.argmax()
    thres_value = thresholds[thres_ind]

    return array_out, thres_value, thres_ind, ((all_prediction_masks, all_temp_masks, all_bounding_box_infos))

class PL_fixed_internal:
    def __init__(self,
                 min_target_scale_inverse=8,
                 mask_rotation=True,
                 merge_close_anomalies=True,
                 anomaly_likelihood_definitely_increasing=False, 
                 save_calculation_data = True,
                 ):
        self.dictionary = {}
        self.min_target_scale_inverse = min_target_scale_inverse
        self.mask_rotation = mask_rotation 
        self.merge_close_anomalies = merge_close_anomalies
        self.anomaly_likelihood_definitely_increasing = anomaly_likelihood_definitely_increasing
        self.save_calculation_data = save_calculation_data

    def __call__(self, heatmaps, targets, 
                 IoU_limit, 
                 key = "0",
                 n_thresholds=25,
                 overlap_limit=1/3):
        
        out = {}
        further_info = {}

        dict_key = f"{self.min_target_scale_inverse}_{n_thresholds}_rot{int(self.mask_rotation)}_mergecloseanom{int(self.merge_close_anomalies)}"
        if dict_key not in self.dictionary:
            data_out_anomalies_increasing = PL_family_rotation_fixed(heatmaps, 
                                                        targets,
                                                        min_target_scale=self.min_target_scale_inverse, 
                                                        n_thresholds=n_thresholds,
                                                        rotation=self.mask_rotation,
                                                        merge_close_anomalies=self.merge_close_anomalies,
                                                        overlap_limit=overlap_limit,
                                                        )
            score_anomales_increasing = (data_out_anomalies_increasing[0]>IoU_limit).mean(axis=1).max()

            if not self.anomaly_likelihood_definitely_increasing:
                data_out_anomalies_decreasing = PL_family_rotation_fixed(-1*heatmaps, 
                                                targets,
                                                min_target_scale=self.min_target_scale_inverse, 
                                                n_thresholds=n_thresholds,
                                                rotation=self.mask_rotation,
                                                merge_close_anomalies=self.merge_close_anomalies,
                                                overlap_limit=overlap_limit,)
                score_anomales_decreasing = (data_out_anomalies_decreasing[0]>IoU_limit).mean(axis=1).max()

                if score_anomales_increasing>score_anomales_decreasing:
                    score = score_anomales_increasing
                    array = data_out_anomalies_increasing[0]
                    data_out_pl = data_out_anomalies_increasing
                else:
                    score = score_anomales_decreasing
                    array = data_out_anomalies_decreasing[0]
                    data_out_pl = data_out_anomalies_decreasing
            else:
                array = data_out_anomalies_increasing[0]
                data_out_pl = data_out_anomalies_increasing

            self.dictionary[dict_key] = array

            if self.save_calculation_data:
                self.dictionary[dict_key+"_data"] = (data_out_pl[1], data_out_pl[2], data_out_pl[3]) 
            else:
                self.dictionary[dict_key+"_data"] = (None, None) 
        else:
            array = self.dictionary[dict_key]
            score = (array>IoU_limit).mean(axis=1).max()
            
            data_out_threshold_masks = self.dictionary[dict_key+"_data"] 
            data_out_pl = (array, data_out_threshold_masks[0], data_out_threshold_masks[1], data_out_threshold_masks[2])

        return score, data_out_pl
                    
    def reset(self):
        self.dictionary = {}

class DataHolderPL_fixed:
    def __init__(self):
        self.dictionary = {}

    def __call__(self, heatmap_set, loss_set, targets_set, paths_set, 
                        IoU_limit, min_target_scale=8, n_thresholds=25,
                         mask_rotation=True, merge_close_anomalies=True, overlap_limit=1/3):
        
        targets = targets_set["targets_novel"]
        out = {}
        further_info = {}
        for key, heatmaps in heatmap_set['heatmap_set_novel'].items():
            dict_key = f"{key}_{min_target_scale}_{n_thresholds}_rot{int(mask_rotation)}_mergecloseanom{int(merge_close_anomalies)}"
            
            score, data_out = PL_fixed_internal()(heatmaps[:,0], targets[:,0], IoU_limit)

            out[key] = score
            further_info[key] = data_out
            # if dict_key not in self.dictionary:
            #     array, _ = PL_family_rotation(heatmaps[:,0], 
            #                                   targets[:,0],
            #                                   min_target_scale=min_target_scale, 
            #                                   n_thresholds=n_thresholds,
            #                                   rotation=mask_rotation,
            #                                   merge_close_anomalies=merge_close_anomalies,
            #                                   overlap_limit=overlap_limit,
            #                                  )
            #     self.dictionary[dict_key] = array

            #     array_rev, _ = PL_family_rotation(-1*heatmaps[:,0], 
            #                                       targets[:,0],
            #                                       min_target_scale=min_target_scale, 
            #                                       n_thresholds=n_thresholds,
            #                                       rotation=mask_rotation,
            #                                       merge_close_anomalies=merge_close_anomalies,
            #                                       overlap_limit=overlap_limit)
            #     self.dictionary[dict_key+"_rev"] = array_rev
            # else:
            #     array = self.dictionary[dict_key]
            #     array_rev = self.dictionary[dict_key+"_rev"]
            # if version=="st":
            #     score = (array>IoU_limit).mean(axis=1).max()
            #     score_rev = (array_rev>IoU_limit).mean(axis=1).max()
            # else:
            #     score = (array.max(axis=0)>IoU_limit).mean()
            #     score_rev = (array_rev.max(axis=0)>IoU_limit).mean()
            # best = np.argmax([score, score_rev])   
            # out[key] = float([score, score_rev][best]) 
            # further_info[key] = [list(row) for row in [array, array_rev][best]]
        
        return out, further_info
            
    def reset(self):
        self.dictionary = {}
        
data_holder_PL_old = DataHolderPL_old()
data_holder_PL = DataHolderPL()
data_holder_PL_fixed = DataHolderPL_fixed()

metric_list = {"Imagewise_AUC": imagewise_AUC, 
               "Pixelwise_AUC": partial(pixelwise_AUC, novel_only=False),
               "Pixelwise_AUC_anom_only": partial(pixelwise_AUC, novel_only=True),
               "Pixelwise_AUPRO": pixelwise_AUPRO,
               "PL_old"   : partial(data_holder_PL_old, IoU_limit=0.3, version="st"),
               "PL_old50"   : partial(data_holder_PL_old, IoU_limit=0.5, version="st"),
               "PL": partial(data_holder_PL, 
                             version="st", 
                             IoU_limit=0.3,
                             mask_rotation=True, 
                             merge_close_anomalies=True),
               "PL50": partial(data_holder_PL, 
                               version="st", 
                               IoU_limit=0.5,
                               mask_rotation=True, 
                               merge_close_anomalies=True),
               "PL_no_merge": partial(data_holder_PL, 
                                      version="st", 
                                      IoU_limit=0.3,
                                      mask_rotation=True, 
                                      merge_close_anomalies=False),
               "PL_original_fixed": partial(data_holder_PL, 
                                            version="st", 
                                            IoU_limit=0.3,
                                            mask_rotation=False, 
                                            merge_close_anomalies=False),
               "PL_original_fixed_merge": partial(data_holder_PL, 
                                                  version="st", 
                                                  IoU_limit=0.3,
                                                  mask_rotation=False, 
                                                  merge_close_anomalies=True),
                "PL_fixed": partial(data_holder_PL_fixed, 
                                    IoU_limit=0.3,),
               #"stPL_50"   : partial(data_holder_PL, IoU_limit=0.5, version="st"),
               #"vtPL_15"   : partial(data_holder_PL, IoU_limit=0.15, version="vt"),
               #"vtPL_30"   : partial(data_holder_PL, IoU_limit=0.3, version="vt"),
               #"vtPL_50"   : partial(data_holder_PL, IoU_limit=0.5, version="vt"),
               }

## this allows certain metrics to be calculated asynchronous in the groups shown below 
metric_key_list_async  = [["Imagewise_AUC", "Pixelwise_AUC"],
                            ["Pixelwise_AUPRO"],
                            ["Pixelwise_AUC_anom_only",
                             "PL",],]

def reset_all():
    data_holder_PL_old.reset()
    data_holder_PL.reset()
    data_holder_PL_fixed.reset()

