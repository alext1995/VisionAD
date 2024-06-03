params = {}

params["results_path"]="test_results" #, type=str)
params["gpu"]="cpu" #, type=int, default=[0], multiple=True, show_default=True)
params["seed"]=0 #, type=int, default=0, show_default=True)
params["log_group"]="group" #, type=str, default="group")
params["log_project"]="project" #, type=str, default="project")
params["save_segmentation_images"]=True #, is_flag=True)
params["save_patchcore_model"]=True #, is_flag=True)
params["backbone_names"]=["wideresnet50"] #, "-b", type=str, multiple=True, default=[])
params["layers_to_extract_from"]=["layer2", "layer3"] #, "-le", type=str, multiple=True, default=[])
# Parameters for Glue-code (to merge different parts of the pipeline.
params["pretrain_embed_dimension"]=1024 #, type=int, default=1024)
params["target_embed_dimension"]=1024 #, type=int, default=1024)
params["preprocessing"]="mean" #, type=click.Choice(["mean", "conv"]), default="mean")
params["aggregation"]="mean" #, type=click.Choice(["mean", "mlp"]), default="mean")
# Nearest-Neighbour ]#Anomaly Scorer parameters.
params["anomaly_scorer_num_nn"]=5 #, type=int, default=5)
# Patch-parameters.
params["patchsize"]=3 #, type=int, default=3)
params["patchscore"]="max" # , type=str, default="max")
params["patchoverlap"]=0.0 #, type=float, default=0.0)
params["patchsize_aggregate"]=[] #, "-pa", type=int, multiple=True, default=[])
# NN on GPU.
params["faiss_on_gpu"]=False #, is_flag=True)
params["faiss_num_workers"]=8 #, type=int, default=8)

params["name"]="mvtec" #, type=str)
params["percentage"]=0.1 #, "-p", type=float, default=0.1, show_default=True)

params["train_val_split"]=1 #, type=float, default=1, show_default=True)
params["batch_size"]=2 #, default=2, type=int, show_default=True)
params["num_workers"]=8 #, default=8, type=int, show_default=True)
params["resize"]=256 #, default=256, type=int, show_default=True)
params["imagesize"]=224 #, default=224, type=int, show_default=True)
params["augment"]=True #, is_flag=True)
params["sampler_name"]="approx_greedy_coreset"