from __future__ import print_function
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from data import *
from model import *
import matplotlib.pyplot as plt



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')
myGene = trainGenerator(2, 'C:/dida_test_task', 'images', 'labels', data_gen_args, save_to_dir=None,image_color_mode="rgb",target_size=(256, 256))
genAug = trainGenerator(1, 'C:/dida_test_task', 'imagesval', 'labelsval', data_gen_args, save_to_dir=None,image_color_mode="rgb",target_size=(256, 256))

model = unet(pretrained_weights="C:/Users/lis/PycharmProjects/dida/3/unet_ThighOuterSurfaceval.hdf5")
testGene = testGenerator("C:/dida_test_task/test2")
results = model.predict_generator(testGene, 5, verbose=1)
saveResult("C:/results2", results)

'''
gen=testGenerator("C:/dida_test_task/test2")

for i in gen:
    from PIL import Image
    import numpy as np
    print(np.size(i))

    import numpy as np
    import matplotlib.pyplot as plt

    plt.imshow(i[0,:,:,:])
    plt.show()

    plt.imshow(np.reshape(i[0][0,:,:,:], (256, 256, 3)))
    plt.show()
    plt.imshow(np.reshape(i[1][0,:,:,:], (256, 256)), cmap='gray')
    plt.show()
'''
model = unet(pretrained_weights="C:/Users/lis/PycharmProjects/dida/3/unet_ThighOuterSurfaceval.hdf5")
model_checkpoint = ModelCheckpoint('unet_ThighOuterSurfaceval.hdf5', monitor='val_loss',verbose=1 , save_best_only=True)
model_checkpoint2 = ModelCheckpoint('unet_ThighOuterSurface.hdf5', monitor='loss',verbose=1, save_best_only=True)


history= model.fit_generator(myGene,validation_data=genAug, validation_steps=3, steps_per_epoch=25, epochs=200, callbacks=[model_checkpoint,model_checkpoint2])

print(history.history.keys())
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'validation loss'], loc='upper left')
plt.savefig('loss.png')

plt.figure()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model loss')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train accuracy', 'validation accuracy'], loc='upper left')
plt.savefig('accuracy.png')
model = unet(pretrained_weights="C:/Users/lis/PycharmProjects/dida/3/unet_ThighOuterSurface.hdf5")
testGene = testGenerator("C:/dida_test_task/test2")
results = model.predict_generator(testGene, 5, verbose=1)
saveResult("C:/results2", results)