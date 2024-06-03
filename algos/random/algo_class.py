from algos.model_wrapper import ModelWrapper
import numpy as np

class WrapperRandom(ModelWrapper):
    def __init__(self, load_params=True, **kwargs):
        super(ModelWrapper, self).__init__()
        if load_params:
            self.load_model_params(**kwargs)
        self.default_segmentation = "method_1"
        self.default_score = "method_1"
        
    def load_model_params(self, **params):
        self.__name__ = "Random"
        self.params = params
    
    def train_one_epoch(self,):
        # self.dataloader_train - use this
        pass
    
    def eval_outputs_dataloader(self, dataloader, len_dataloader):
        length = 0
        for item, _ in dataloader:
            length+=item.shape[0]

        a_pixel_ret  = {"method_1":np.random.rand(length, 256, 256), 
                        "method_2":np.random.rand(length, 256, 256)}
        a_image_ret  = {"method_1":np.random.rand(length),
                        "method_2":np.random.rand(length)}
        return a_pixel_ret, a_image_ret
    
    def pre_eval_processing(self):
        pass
        
    def save_model(self, location):
        pass
    
    def load_model(self, location):
        pass