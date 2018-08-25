from keras.optimizers import Adam

from core.decay import *


# TODO: Config serialization/deserialization + change settings via parameters
from core.losses.loss_terms_base import AdversarialLoss


class Config:
    def __init__(self):
        #######################
        # Training papameters #
        #######################
        self.n_epochs = 200
        self.batch_size = 1
        self.load_checkpoint = False
        self.snapshot_interval = 500
        self.checkpoint_interval = 20

        ###################
        # Hyperparameters #
        ###################
        self.image_shape = (128, 128, 3)
        self.channels = self.image_shape[2]

        # Layer parameters
        self.use_resize_convolution = False
        self.use_spectral_normalization = False
        self.gen_filters = 32
        self.dis_filters = 64
        self.residual_blocks = 6 if self.image_shape[1] <= 128 else 9
        self.dropout_rate = 0.5
        self.pool_size = max(self.batch_size, 50)
        self.n_critic = 5  # Only used with wasserstein loss

        # Loss
        self.adversarial_loss = AdversarialLoss.LSGAN
        self.use_cycle_loss = True
        self.use_identity_loss = False

        self.optimizer_g = Adam(lr=0.0, beta_1=0.5, beta_2=0.9)  # learning rate is overwritten by the schedule
        self.optimizer_d = Adam(lr=0.0, beta_1=0.5, beta_2=0.9)

        self.lr_g_schedule = LinearSchedule(0.0002, 0, 100, 200)
        self.lr_d_schedule = LinearSchedule(0.0002, 0, 100, 200)
        self.l_dis_schedule = IdentitySchedule(value=1)
        self.l_cycle_schedule = IdentitySchedule(value=10)
        self.l_id_schedule = IdentitySchedule(value=0.5)
        self.l_gp_schedule = IdentitySchedule(value=10)

        ###########
        # Folders #
        ###########
        # Training
        self.train_dataset_dir = r'../resources/input/dataset/'
        self.snapshot_dir = f'../resources/output/snapshots/'
        self.checkpoints_dir = r'../resources/output/checkpoints/'
        self.log_dir = r'../resources/output/logs/'

        self.dataset_name = r'cityscapes_jpg'
        self.folder_train_a = r'trainA'
        self.folder_train_b = r'trainB'
        self.folder_test_a = r'testA'
        self.folder_test_b = r'testB'

        # Predict
        self.predict_input_dir = r'../resources/input/predict/'
        self.predict_output_dir = r'../resources/output/predict/'
