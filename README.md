# CycleGAN Keras
Implementation of CycleGAN in Keras. 

To train:
```
python main.py train
```

To test:
```
python main.py predict
```

For now just edit ./source/util/config.py to change hyperparameters. You can change the location of the folders in config.py as well. By default, the training dataset should be in ./resources/input/dataset/<my_dataset>/. A training dataset should contain a 'trainA', 'trainB', 'testA' and 'testB' subfolder, but that can be changed in config.py too.
By default, the test images should be in ./resources/input/predict/.

Logs are saved to ./resources/output/logs/ for tensorboard and in plain csv.

If you want to create a new loss term, create a new subclass of LossTerm in ./source/core/losses/ and add it in the initialize_loss_terms() function in cyclegan.py.

The models (residual generator, u-net generator, patchgan discriminator) are in ./source/core/models/. You can change which kind of model to use for the discriminator or generator in main.py. 

Also implemented:
* Relativistic average LSGAN
```python
# in config.py
self.adversarial_loss = AdversarialLoss.RLSGAN 
```

* Spectral normalization
```python
# in config.py
self.use_spectral_normalization = True
```

* Wasserstein loss (CycleGAN does not seem to work properly with it)
```python
# in config.py
self.adversarial_loss = AdversarialLoss.WGAN  
```

* Decay schedules for hyperparameters (linear decay, warm restarts and 'identity' schedule)
```python
# in config.py
self.lr_g_schedule = LinearSchedule(start_value=0.0002, end_value=0, start_epoch=100, end_epoch=200) 
```

