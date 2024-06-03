from algos.fastflow2d._defaultconfigs import fastflow_default_model_params
image_size = 256

## different modes: 
# single_channel_mean_only
# single_channel_mean_var
# multi_channel_mean_only
# multi_channel_mean_var

ff_altub_default_model_params = {"fastflow_params": fastflow_default_model_params,
                                 "freezing_interval": 5,
                                 "mode": "multi_channel_mean_var",
                                 "input_size": image_size,
}