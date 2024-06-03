ddad_default_model_params = {"input_size": 256,
          "batch_size": 32,
          "input_channel": 3,
          "feature_extractor": "wide_resnet101_2",  # wide_resnet50_2 #resnet50
          "learning_rate": 3e-4, 
          "weight_decay": 0.05, 
          "epochs": 2000,
          "load_chp" : 1000, # From this epoch checkpoint will be loaded. Every 250 epochs a checkpoint is saved. Try to load 750 or 1000 epochs for Visa and 1000-1500-2000 for MVTec.
          "DA_epochs": 3,  # Number of epochs for Domain adaptation. Try [0-3]. "Fine tuninig this parameter results in better performance".
          "eta" : 1, # Stochasticity parameter for denoising process.
          "v" : 7, # Control parameter for pixel-wise and feature-wise comparison. v * D_p + D_f
          "w" : 4, # Conditionig parameter. The higher the value, the more the model is conditioned on the target image. "Fine tuninig this parameter results in better performance".
          "w_DA" : 3, # Conditionig parameter for domain adaptation. The higher the value, the more the model is conditioned on the target image.
          "trajectory_steps": 1000,
          "test_trajectoy_steps": 200,   # Starting point for denoining trajectory.
          "test_trajectoy_steps_DA": 200,  # Starting point for denoining trajectory for domain adaptation.
          "skip" : 20,   # Number of steps to skip for denoising trajectory.
          "skip_DA" : 20,
          "beta_start" : 0.0001,
          "beta_end" : 0.02,
          }