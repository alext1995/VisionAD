# VisionAD - a library of the most performant anomaly detection algorithms

[Published at TMLR](https://openreview.net/forum?id=o5kYH7bNe3)

VisionAD contains the largest and most performant collection of anomaly detection algorithms. All algorithms are written in a standardised manner such that they can be called by the same API. The library has a focus on:

1. Benchmarking - as the algorithms are called and evaluated with the same code, the library can undertake a fair benchmarking of the currently available anomaly detection algorithms. This mitigates the issue of cherry-picked results. 

2. Rapid experimentation - as the data loading and evaluation code are handled, the reseacher can dive straight into algorithm development. If the algorithm is written in the standardised format (as shown in the provided notebook), it fits straight into the library (see the section on adding algorithms).

## Getting started

1. Clone the repository
2. `pip3 install -r requirements.txt` / `pip3 install -r requirements_all.txt`
3. Download the [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad/)/[VisA](https://github.com/amazon-science/spot-diff?tab=readme-ov-file#data-download) datasets
4. Fill the paths to the MVTec and VisA datasets in `data/configure_dataset.py`
5. Run one of the algorithms to check that everything is working:

```
python3 run.py --config configs/ppdm/ppdm_mvtec.py --wandb 0 --device "cuda:0"
```

## Command line use

The following line is sufficient to load a config file and run the algorithms and datasets specified.

```
python3 run.py --config configs/patchcore/patchcore_mvtec.py --run_description "patchcore_demo" --device "cuda:0" --wandb 0
```

```
python3 run.py --config configs/{my_config_file} --run_description "my_experiment" --device "cuda:0"
```

## Config files

The config file needs a variable `dataset_list`. This tells which the wrapper which datasets to run. It also needs a list a of metrics to run. Setting the first item to 'all' will mean all metrics are ran. 

```
from torchvision import transforms
from PIL import Image
dataset_list = ['mvtec_bottle',
                 'visa_pcb4']
                 
metrics = ["all"]
```

The config also needs a `model_list`. Each item in the list is a dictionary, where the dictionary tells the wrapper the algorithm, hyperparameters, data loading parameters, and evaluation parameters. 

```
from algos.cfa._defaultconfigs import cfa_default_model_params
model_list = []
model_list.append({"algo_class_name": "cfa", # algorithm, see algos/model_wrapper.py for key-class dictionary
                    "model_params": cfa_default_model_params, # hyperparameters for algorithm
                    "epochs": cfa_default_model_params["epochs"], # or enter the number
                    "training_transformations":  transforms.Compose([]), # specific transformations to run on training data
                    "training_transformation_string": "transforms.Compose([])", # string of these transformations (for logging)
                    "identical_transforms": transforms.Compose([transforms.Resize((image_size, image_size), 
                                                                interpolation=Image.NEAREST),
                                                                ]), # transformations to run on training and testing data
                    "batch_size": 4, # batch size 
                    "save_model_every_n_epochs": None, # saves model every n epochs 
                    "save_heatmaps_n_epochs": None, # saves the predictions of the model every n epochs
                    "evaluate_n_epochs": 2, # evaluates the algorithm every n epochs
                    "test_batch_size": 8, # test batch size
                    "device": "cuda:0", # device (overwritten if device specificed in command line)
                    "train_time_limit": 2*3600, # training time limit in seconds, after this training stops (note the timer does not include inference during testing or calculating of metrics)
                    "input_size": 256, # input image size
                    "model_description": "CFA_mvtec_run", # description saved to each run from the dictionary - DIFFERENT to run_description in the cmd, which is the description attached to all the runs in a given config file - use no more than 20 characters - used in the results directory name
                    "model_long_description": "Feel free to put a longer description of this model in here if you desire. This will be saved to the WandB run, the saved predictions, and a json file in the results directory of the run.", 
                    "save_metric_meta_data": False, # whether the metric data is saved alongside the metric results, setting as True will result in a lot of physical memory consumption
                    "wandb": True, # upload the data to Wandb or not
                    }) 
#model_list.append({...})
```

The `model_list` can contain any number of algorithms/hyperparameter combinations. By default each algorithm in `model_list` is combined with each dataset in `dataset_list`. 

However, you can configure each model to run a bespoke list of datasets by adding the `dataset_list_override` key to the model dictionary: 

```
model_list = []
model_list.append({"algo_class_name": "cfa", # algorithm, see algos/model_wrapper.py for key-class dictionary
                    "model_params": cfa_default_model_params, # hyperparameters for algorithm
                    ...
                    model_arguments["dataset_list_override"] = ['mvtec_bottle', visapcb4'] # with these key present, only these datasets are ran for this model
                    })
```
We choose .py for config files because it enables the user to use the flexibility of Python. For instance, if one wanted to test an algorithm with various hyperparameters, they could create a for loop which dynamically changes the given hyperparameter and description. 

## Adding algorithms

The ease of adding algorithms is intended to be one of the major strengths of VisionAD. A walkthough Ipython file shows the process of adding an algorithm, based on an algorithm which makes random predictions.

Each algorithm is built from a class called `{algo_name}Wrapper`, which inherits the `ModelWrapper` class. Using the provided template, a researcher only needs to implement six methods for the algorithm to work. Each method is discussed below:

Initialisation: 

`__init__(self)`

Initialise the model and optimisers.

Training: 

`train_one_epoch(self)`

Train the model using the self.dataloader_train (automatically attached via the wrapper).

Pre-eval processing: 

`prev_eval_processing(self)`

Optionally undertake any necessary pre-processing before evaluation, using access to self.dataloader_train. Left as `pass' in most algorithms. This cannot be done at the start of eval_outputs_dataloader, as eval_outputs_dataloader may be called on a new class instance in the case of loading the model from disk, where the train generator would not be available. 

Evaluation: 

`eval_outputs_dataloader(self, generator, len_generator)`

This method is called twice, once with a generator of regular test images, and once with a generator of anomalous test images. To avoid data leakage, this method does not have access to ground truths or paths. The method must iterate over the passed generator, saving to memory an anomaly map and score for each image. Returns a tuple of two dictionaries:  a dictionary of anomaly maps: 'anomaly_map_method_a': torch.tensor|np.array of shape (no.images, height, width), and a dictionary image scores torch.tensor|np.array of shape (no.images). Notably, the system allows any number of different methods for creating image scores and heatmaps for a given algorithm. For instance, the researcher may wish to trial reducing a feature channel via mean, standard deviation, and max, without wishing to rewrite three algorithms. The metric code will provide results for each key and tensor passed through. The code for demo random algorithm should make this clear. 

Saving: 

`save_model(self, location)`

Saves the parameters of the model to a given location.

Loading: 

`load_model(self, location)`

Load the parameters of the model from a given location.

See the trivial example of a random classifier, and the Ipython file in the notebooks folder. Amend this code as you wish, as long as it uses the API structure given, and outputs predictions in the right shape/format, the code will fit into the library.

## Ipython experimentation
Enabling easy experimentation is a goal of VisionAD. Early stage algorithm development is often done in an Ipython environment. The necessary template code can be loaded into a Ipython environment such as Jupyter notebook. The researcher can experiment with the algorithm class whilst the data loading and evaluation code is handled. A starter notebook is provided to demonstrate this using the trivial random classifier. 

## Metric format
VisionAD allows algorithms to output variations of their predictions. For instance, an algorithm may output predictions by reducing an array via mean, minumum, standard deviation. These can all be tested without incurring the cost of running the algorithm multiple times/writing different variations of the algorithm. 

The format of the metric files is:

```
{'Imagewise_AUC': {'metric_results': {'algo_variation_1': 0.526984126984127,
                                      'algo_variation_2': 0.5380952380952382},
                   'metric_data':   {'algo_variation_1': ...,
                                     'algo_variation_2': ...}},
 'Pixelwise_AUC': ...,
 'Pixelwise_AUPRO': ...,
 'PL': ...}
```

However, we encourage you to use the WandB integration, as this makes logging and viewing the results much easier.

## Adding datasets

The dataset file datasets/configure_dataset contains a variable called datasets, which is a dictionary of dataset keys, where the item of each key is a dictionary with various dataset information (training image path, testing image path, ...). These should be self explanatory. These dataset keys should match the strings in the run configuration files. To add a dataset a user needs to add a dataset key and fill the necessary paths. See the demo_dataset key below:

```
datasets["demo_dataset"] = {"folder_path_regular_train" : r"",
                            "folder_path_regular_test": r"",
                            "folder_path_novel_test"  : r"",
                            "folder_path_novel_masks"  : r"",
                            "regular_train_start" : None,
                            "regular_train_end"   : None,
                            "regular_test_start"  : None, 
                            "regular_test_end" : None,
                            "novel_test_start" : None, 
                            "novel_test_end" : None,
                            "skip_train_regular" : 1, 
                            "skip_test_regular"  : 1, 
                            "skip_test_novel"    : 1, 
                            "mean"         : [0.485, 0.456, 0.406],
                            "std"          : [0.229, 0.224, 0.225],
                            "pixel_labels" :  True,
}
```

The names of the items in the novel_mask folder should match this in the novel_test folder, but can end in .png/_mask.png. For instance, if a novel image is called screw1.jpg, the corresponding mask could be called screw1.jpg/screw1.png/screw1_mask.png.

## Synethetic anomalies
In some algorithms, synthetic anomalies are created by adding pertubations to the training images, and using the information relating to this pertubation in training (i.e. masks). This is achieved in VisionAD by passing a callback to the train_dataloader as follows:

```
def callback_function(pil_image):
    # do something to the pil_image and apply 
    image = transforms.ToTensor()(image)

    pertubed_image = image[100:100, 100:100] * 2

    # you can send anything which can be batched by a Pytorch dataloader
    pertubation_meta_information = 2

    # usually this is used to apply noise to an image, and the meta information keeps record of where the noise was added 

    return pertubed_image, pertubation_meta_information

dataloader_train.dataset.pre_normalisation_transform = callback_function

for pertubed_image, path, pertubation_meta_information in dataloader_train:
    # pass
```

## Other features
As mentioned above, the library does not give algorithms access to the ground truths during training, to ensure data leakage is not possible. The wrapper also contains a number of other bug checking features such as ensuring the outputs of the algorithms are the right dimensions. The wrapper also measures total training time, training time per image, total inference time, and inference time per image. The wrapper allows all metrics to be logged to Weights and Biases (Wandb) if desired, and code is also provided to pull these results from Wandb and parse them into the tables shown in this publication.   

Some algorithms require synthetic anomalies. To facilitate this, we allow a callback to be ran on the training data before it is sent through the dataloader. The training dataloader outputs three items, the training image, the result of this callback, and the image file name. The default of the callback result is to return 0, which would be passed to the algorithm and ignored. However in the case that synthetic anomalies are added to the training images, this callback could return the corresponding mask, which the algorithm can use.

## Packages

The base packages needed to run the library are shown below (included in the requirements.txt file)
```
pip3 numpy
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install opencv-python
pip3 install scikit-learn
pip3 install scikit-image
pip3 install shapely
pip3 install wandb
pip3 install tqdm
pip3 install scipy
pip3 install proportion-localised
```

The list of packages will ensure the following algorithms run:

PPDM
PEFM
PFM
CDO
Reverse distillation

A longer set of requirements, which allow all the algorithms to run can be found in requirements_all.txt

```
pip3 install einops
pip3 install efficientnet-pytorch
pip3 install FrEIA
pip3 install kornia
pip3 install timm
pip3 install imgaug
pip3 install faiss-gpu
```

If you only wish to install the base packages and a few extra per algorithm, below are the extra packages needed per algorithm:


```
Patchcore
pip3 install timm
pip3 install faiss-gpu

Fastflow
pip install timm
pip install FrEIA

Msflow
pip3 install FrEIA

DDad
pip install kornia

Simplenet
pip3 install timm

Cflow
pip install timm
pip install FrEIA

CFA
pip3 install einops

AST
pip3 install efficientnet-pytorch
pip3 install FrEIA 

Memseg
pip3 install timm
pip3 install imgaug

Efficientad
Requires downloading these models and entering their paths:
/models/teacher_medium.pth
/models/teacher_small.pth
```

# Transactions of Machine Learning Research (TMLR) Publication
More details can be found in the Transactions of Machine Learning Research (TMLR) publication: [VisionAD, a software package of performant anomaly detection algorithms, and Proportion Localised, an interpretable metric](https://openreview.net/forum?id=o5kYH7bNe3)
