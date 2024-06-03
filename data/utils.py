import os
import shutil

def add_dataset(datasets, path, name, skip_train_regular=1, skip_test_regular=1, skip_test_novel=1):
    datasets[name] = {"dataset_key": name,
                    "folder_path_regular_train" : os.path.join(path, "train", "good"),
                    "folder_path_regular_test"  : os.path.join(path, "regular_test_grouped"),
                    "folder_path_novel_test"  : os.path.join(path, "anomaly_test_grouped"),
                    "folder_path_novel_masks" : os.path.join(path, "ground_truth_grouped"),
                    "regular_train_start" : None,
                    "regular_train_end" : None,
                    "regular_test_start" : None, 
                    "regular_test_end" : None,
                    "novel_test_start" : None, 
                    "novel_test_end" : None,
                    "skip_train_regular" : skip_train_regular, 
                    "skip_test_regular"  : skip_test_regular, 
                    "skip_test_novel"    : skip_test_novel, 
                    "mean"         : [0.485, 0.456, 0.406],
                    "std"          : [0.229, 0.224, 0.225],
                    "pixel_labels" :  True,
                    }
    return datasets

def configure_mvtec_class(class_path):
    test_grouped         = os.path.join(class_path, "anomaly_test_grouped")
    ground_truth_grouped = os.path.join(class_path, "ground_truth_grouped")
    os.makedirs(test_grouped, exist_ok=True)
    os.makedirs(ground_truth_grouped, exist_ok=True)
    for class_ in os.listdir(os.path.join(class_path, "test")):
        if class_=="good":
            continue
        for file in os.listdir(os.path.join(class_path, "test", class_)):
            source = os.path.join(class_path, "test", class_, file)
            dest   = os.path.join(test_grouped, class_+"_"+file)
            shutil.copy(source, dest)
            
            source = os.path.join(class_path, "ground_truth", class_, file[:-4]+"_mask.png")
            dest   = os.path.join(ground_truth_grouped, class_+"_"+file[:-4]+"_mask.png")
            shutil.copy(source, dest)


