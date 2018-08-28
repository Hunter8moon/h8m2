# CycleGAN Keras
Implementation of CycleGAN in Keras. 

To test:
```
 $ python main_test.py <dir_input> <dir_output> <filename_model>
```


To train:
```
 $ python main_train.py <dir_input>
```

For now just edit ./source/util/config.py to change hyperparameters.


Folders (can all be changed in config.py):
* Input: 
   * By default, the training dataset should be in './resources/input/<name_dataset>/'. 
   * A training dataset should contain a 'trainA', 'trainB', 'testA' and 'testB' subfolder.

* Output:
   * Logs are saved to './resources/output/<name_dataset>/logs/' for tensorboard and in plain csv.
   * Checkpoints are saved to './resources/output/<name_dataset>/checkpoints/' in .h5 files.
   * Snapshots are saved to './resources/output/<name_dataset>/snapshots/'.

If you want to create a new loss term, create a new subclass of LossTerm in ./source/core/losses/ and add it in the initialize_loss_terms() function in cyclegan.py.

The models (residual generator, u-net generator, patchgan discriminator) are in ./source/core/models/. You can change which kind of model to use for the discriminator or generator in main.py.
You can load the latest checkpoint by setting 
```python
# in config.py
self.load_checkpoint = True
``` 
This will load the models of the latest epoch from './resources/output/<name_dataset>/checkpoints/'.

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

* Wasserstein loss (does not seem to work properly)
```python
# in config.py
self.adversarial_loss = AdversarialLoss.WGAN  
```

* Decay schedules for hyperparameters (linear decay, warm restarts and 'identity' schedule)
```python
# in config.py
self.lr_g_schedule = LinearSchedule(start_value=0.0002, end_value=0, start_epoch=100, end_epoch=200) 
```

