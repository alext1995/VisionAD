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
        for image, path, image_callback_info in self.dataloader_train:
            pass 
            
        ## image_callback_info is only relevant if you add a callback to the 
        ## dataloader_train. If you don't, you can ignore it
        ## see the Synthetic anomalies section in the Readme for more information
        
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
        self.save_params(location) # necessary code 
        # save other stuff here such as ml weight, e.g.
        # torch.save(self.ml_model, os.path.join(location, "model.pt"))
        
    def load_model(self, location):
        params = self.load_params(location)  # necessary code 
        self.load_model_params(**params)  # necessary code 
        # load other stuff here such as ml weight, e.g.
        # self.ml_model = torch.load(os.path.join(location, "model.pt"))