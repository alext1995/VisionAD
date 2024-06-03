img_len = 768
img_size = [192, 192]
depth_downscale = 8
n_coupling_blocks =  4
batch_size = 8

ast_default_model_params = {"img_len" : 768, # width/height of input image
    "input_size": img_len,
    "n_feat": 304,
    "img_size" : (img_len, img_len), 
    "img_dims" : [3] + list(img_size),
    "depth_len" : img_len // 4, # width/height of depth maps
    "depth_downscale" : depth_downscale, # to which factor depth maps are downsampled by unshuffling
    "depth_channels" : depth_downscale ** 2, # channels per pixel after unshuffling
    "map_len" : img_len // 32, # feature map width/height (dependent on feature extractor!)
    "extract_layer" : 35, # layer from which features are extracted
    "img_feat_dims" : 304, # number of image features (dependent on feature extractor!)
    "clamp" : 1.9, # clamping parameter
    "n_coupling_blocks" : 4, # higher = more flexible = more unstable
    "channels_hidden_teacher" : 64, # number of neurons in hidden layers of internal networks
    "channels_hidden_student" : 1024, # number of neurons in hidden layers of student
    "use_gamma" : True,
    "kernel_sizes" : [3] * (n_coupling_blocks - 1) + [5],
    "pos_enc" : True, # use positional encoding
    "pos_enc_dim" : 32, # number of dimensions of positional encoding
    "asymmetric_student" : True,
    "n_st_blocks" : 4, # number of residual blocks in student
    "lr" : 2e-4, # learning rate
    "batch_size" : 8,
    "eval_batch_size" : batch_size * 2,
    "hide_tqdm_bar" :  False,      
    "dilate_mask" : True,
    "n_fills" : 3,
    "bg_thresh" : 7e-3,
    "pre_extracted": False,
    'mode': 'RGB',
    'eval_mask': True,
    "switch_teacher_student_training_epoch": 72,
    }