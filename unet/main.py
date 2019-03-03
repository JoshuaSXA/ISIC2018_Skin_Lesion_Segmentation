from unet.res_unet import *
from unet.utils import *
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.callbacks import TensorBoard
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     shear_range=0.1,
                     zoom_range=0.1,
                     horizontal_flip=True,
                     # vertical_flip=True,
                     fill_mode='nearest',
                    )

myGene = trainGenerator(2,'./data/train_aug/', 'image','mask',data_gen_args,save_to_dir = None)
valGene = valGenerator(1, './data/val/', 'image', 'mask')

model = res_unet()

weight_path = "./saved_model/resunet_jaccloss_dice_jacc_0.66645.hdf5"
model = load_model(weight_path, custom_objects={'bce_jacc_loss': bce_jacc_loss, 'dice_coef': dice_coef, 'jacc_coef': jacc_coef})

logs_path = './graph/log_bcejaccloss_aug'

tbCallBack = TensorBoard(log_dir=logs_path)

cb = [ModelCheckpoint('resunet_jaccloss_dice_jacc_dataaug.hdf5', monitor='val_jacc_coef', mode = 'max',verbose=1, save_best_only=True),
      EarlyStopping(monitor='val_jacc_coef',patience=40, verbose=2, mode='max'),
      ReduceLROnPlateau(monitor='val_jacc_coef', factor=0.1, patience=5,  mode='max', epsilon=0.0001),
      tbCallBack]

model.fit_generator(myGene,steps_per_epoch=8300, epochs=100,callbacks=cb, validation_data=valGene, validation_steps=519)


num_image = len(glob.glob(os.path.join('./data/org_data/test', '*')))
valGene = commitValGenerator('./data/org_data/test')

get_val_res(model, valGene, num_image, save_path="./data/66645test/")
