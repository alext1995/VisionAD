cfa_default_model_params = {'Rd': False, #default=False)
                            'cnn': "wrn50_2", #choices=['res18', 'wrn50_2', 'effnet-b5', 'vgg19'], default='wrn50_2')
                            'gamma_c': 1, #default=1
                            'gamma_d': 1, #default=1
                            'lr'     : 1e-3,
                            'weight_decay' : 5e-4,
                            'amsgrad'      : True,
                            'input_size'   : 256,
                            'epochs'       : 30,
                            }