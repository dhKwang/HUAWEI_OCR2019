
from model import *
from data import *
from keras import backend as K
import tensorflow as tf

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
sess = tf.Session(config=config)
K.set_session(sess)
data_gen_args = dict(rotation_range=0.2,##
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

BATCH_SIZE=2
STRPS_PER_EPOCH=60
EPOCHS=25

myGene = trainGenerator(BATCH_SIZE,'F:/data/汉字识别',\
    'ORI_IMAGE','MASK_IMAGE',data_gen_args,save_to_dir = None,target_size=(256,256))


model = unet(input_size=(256,256,1))

cp_fn='F:/GitHub Code/unet-master/OCR_pad_0322_{}{}{}.hdf5'.\
    format(BATCH_SIZE,STRPS_PER_EPOCH,EPOCHS)
model_checkpoint = ModelCheckpoint(cp_fn\
    , monitor='loss',verbose=1, save_best_only=True)


model.fit_generator(myGene,\
    steps_per_epoch=STRPS_PER_EPOCH,epochs=EPOCHS,callbacks=[model_checkpoint])

testGene = testGenerator("F:/data/汉字识别/TEST",num_image=2,target_size=(256,256))
results = model.predict_generator(testGene,2,verbose=1)
print('BATCH_SIZE:{},STRPS_PER_EPOCH:{}，EPOCHS:{}'.format(BATCH_SIZE,STRPS_PER_EPOCH,EPOCHS))
# saveResult("F:/data/汉字识别/PREDICT",results)