from model import *
from data import *
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')

myGene = trainGenerator(2,'../unet/data/ISIC2018/train/', 'image','mask',data_gen_args,save_to_dir = None)
valGene = valGenerator(2, '../unet/data/ISIC2018/val/', 'image', 'mask')

model = unet()
weight_path = "./unet_membrane.hdf5"
model = load_model(weight_path, custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef})
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=640,epochs=100,callbacks=[model_checkpoint], validation_data=valGene, validation_steps=30)


# results = model.predict_generator(testGene,30,verbose=1)
# saveResult("./checkpoints/",results)

num_image = len(glob.glob(os.path.join('../unet/data/ISIC2018/test/image', '*')))

testGene = testGenerator('../unet/data/ISIC2018/test/', num_image)

get_test_res(model, testGene)