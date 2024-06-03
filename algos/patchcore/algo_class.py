'''
Based on code found here: https://github.com/amazon-science/patchcore-inspection
And based on the the paper: 
https://arxiv.org/pdf/2106.08265v2.pdf
@misc{roth2021total,
      title={Towards Total Recall in Industrial Anomaly Detection},
      author={Karsten Roth and Latha Pemula and Joaquin Zepeda and Bernhard Schölkopf and Thomas Brox and Peter Gehler},
      year={2021},
      eprint={2106.08265},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
'''
import numpy as np
import torch
import os
from algos.model_wrapper import ModelWrapper
from tqdm import tqdm
from algos.patchcore import backbones
from algos.patchcore import common
from algos.patchcore.patchcore import PatchCore
from algos.patchcore import sampler
from algos.patchcore import utils
import tempfile
import shutil

class WrapperPatchCore(ModelWrapper):
    def  __init__(self, load_model_params=True, **params):
        super(ModelWrapper, self).__init__()
        if load_model_params:
            self.load_model_params(**params)
        self.default_segmentation = "segmentations"
        self.default_score = "scores"


    def load_model_params(self, **params):
        self.__name__ = "PatchCore"
        self.params = params
        sampler = get_sampler(params)
        size = (3,params["input_size"],params["input_size"])
        self.PatchCore_list = patch_core(params)(size, sampler, params["device"])

        for PatchCore in self.PatchCore_list:
            torch.cuda.empty_cache()
            if PatchCore.backbone.seed is not None:
                utils.fix_seeds(PatchCore.backbone.seed, params["device"])

    def save_model(self, location):
        os.makedirs(location, exist_ok=True)
        self.save_params(location)
        os.makedirs(os.path.join(location, "features"), exist_ok=True)
        for i, PatchCore in enumerate(self.PatchCore_list):
            prepend = str(f"ensemble{i}_")
            PatchCore.save_to_path(os.path.join(location, "features"), prepend)

    def load_model(self, location, **kwargs):
        params = self.load_params(location)
        self.load_model_params(**params)

        faiss_on_gpu = False
        faiss_num_workers = 8
        device = params["device"]

        files = os.listdir(os.path.join(location, "features"))
        self.PatchCore_list = []
        for i, file in enumerate(files):
            if not "patchcore_params.pkl" in file:
                continue
            prepend = file.split("_")[0]+"_"
            nn_method = common.FaissNN(faiss_on_gpu, faiss_num_workers)
            patchcore_instance = PatchCore(device)
            patchcore_instance.load_from_path(os.path.join(location, "features"), device, nn_method, prepend=prepend)
            self.PatchCore_list.append(patchcore_instance)

    def train_one_epoch(self):
        '''
        This model actually does no 'training'. It just gathers the features using the pre-trained backbone.
        This action of doing this makes more sense in the pre_eval_processing part
        '''
        pass

    def pre_eval_processing(self,):
        for i, PatchCore in enumerate(self.PatchCore_list):
            PatchCore.fit(self.dataloader_train)

        ''' 
        Save the model
        Reload it with faiss_on_gpu = False
        A bug in faiss means it keeps crashing during inference when a GPU is used. 
        The only way I found around this is to save it and 
        reload it with faiss_on_gpu=False, and device="cpu" 
        '''
        location = os.path.join(tempfile.gettempdir(), f"patchcore_internal_save_point_{str(np.random.rand())[2:]}")
        if os.path.exists(location): 
            shutil.rmtree(location)
        while os.path.exists(location): 
            pass
        self.save_model(location)
        self.load_model(location, faiss_on_gpu=False, faiss_num_workers=8, device="cpu")
        shutil.rmtree(location)
        while os.path.exists(location):
            pass

    def _predict(self, PatchCore, dataloader, len_dataloader, verbose=False, limit=None):
        paths_ret = []
        scores    = []
        for ind, (data, _) in tqdm(enumerate(dataloader), total=len_dataloader):
            torch.cuda.empty_cache()
            _scores, _segmentations = PatchCore._predict(data)
            _scores, _segmentations = _scores.to("cpu"), _segmentations.to("cpu")
            scores += [score.item() for score in _scores]

            if ind==0:
                segmentations_set = _segmentations
            else:
                segmentations_set = torch.vstack([segmentations_set, _segmentations])
            if limit:
                if ind>limit:
                    break
        return scores, segmentations_set

    def eval_outputs_dataloader(self, dataloader, len_dataloader, verbose=False, limit=None):
        aggregator = {"scores": [], "segmentations": []}
        for i, PatchCore in enumerate(self.PatchCore_list):
            
            torch.cuda.empty_cache()
            PatchCore.anomaly_scorer.nn_method.faiss_on_gpu = False            
            scores, segmentationss = self._predict(
                PatchCore,
                dataloader,
                len_dataloader,
                verbose=verbose,
                limit=limit,
            )

            # targets and paths are the same whichever model we extract from
            # therefore we only need to stack scores and segmentations for manipulation later
            aggregator["scores"].append(scores)
            aggregator["segmentations"].append(segmentationss)

            
        scores = np.array(aggregator["scores"])
        # min_scores = scores.min(axis=-1).reshape(-1, 1)
        # max_scores = scores.max(axis=-1).reshape(-1, 1)
        # scores_normalised = (scores - min_scores) / (max_scores - min_scores)
        # scores_normalised = np.mean(scores_normalised, axis=0)

        segmentations = torch.stack(aggregator["segmentations"]).numpy()

        # the normalisation of the segmentations and scores is removed
        # it makes no sense when the regular and anomaly dataloader are evaluated seperately
        segmentations = np.mean(segmentations, axis=0)
        scores        = np.mean(scores, axis=0)

        return {"segmentations":torch.tensor(segmentations).unsqueeze(1)}, {"scores": scores}

def patch_core(params):
    pretrain_embed_dimension = params["pretrain_embed_dimension"]
    target_embed_dimension = params["target_embed_dimension"]
    preprocessing = params["preprocessing"]
    aggregation = params["aggregation"]
    patchsize = params["patchsize"]
    patchscore = params["patchscore"]
    patchoverlap = params["patchoverlap"]
    anomaly_scorer_num_nn = params["anomaly_scorer_num_nn"]
    patchsize_aggregate = params["patchsize_aggregate"]
    faiss_on_gpu = params["faiss_on_gpu"]
    faiss_num_workers = params["faiss_num_workers"]
    
    backbone_names_layers = params["backbone_names_layers"]

    def get_patchcore(input_shape, sampler, device):
        loaded_patchcores = []


        for backbone_name, layers in backbone_names_layers.items():
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                    backbone_name.split("-")[-1]
                )
            backbone = backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed
            nn_method = common.FaissNN(faiss_on_gpu, faiss_num_workers)

            patchcore_instance = PatchCore(device)
            patchcore_instance.load(
                backbone=backbone,
                layers_to_extract_from=layers,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                featuresampler=sampler,
                anomaly_scorer_num_nn=anomaly_scorer_num_nn,
                nn_method=nn_method,
            )
            loaded_patchcores.append(patchcore_instance)
        print("len(loaded_patchcores):", len(loaded_patchcores))
        return loaded_patchcores
    return get_patchcore

def get_sampler(params):
    name = params["sampler_name"]
    percentage = params["percentage"]
    faiss_on_gpu = params["faiss_on_gpu"]

    if faiss_on_gpu:
        device = utils.set_torch_device(params["device"])
    else:
        device = "cpu"

    if name == "identity":
        return sampler.IdentitySampler()
    elif name == "greedy_coreset":
        return sampler.GreedyCoresetSampler(percentage, device)
    elif name == "approx_greedy_coreset":
        return sampler.ApproximateGreedyCoresetSampler(percentage, device)

    raise NotImplemented("sampler_name: '{sampler_name}' not in available sampler names")
