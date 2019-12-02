1. Final Validation accuracy for the base network is 82.68

2. New Model
## New model
weight_decay = 0.00001
model1 = Sequential()
model1.add(SeparableConv2D(filters= 32,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay),depth_multiplier = 1,input_shape=(32,32,3),activation='relu'))  #30
model1.add(BatchNormalization())
model1.add(Dropout(0.2))
# 30*30*32
#RFo=RFi+(k-1)*jin
#Jout=Jin * s
#Jin=1,s=1, RFi=1, k=3, Jout=1
#receptive field = 1+(3-1)*1=3
model1.add(SeparableConv2D(filters= 64,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay),depth_multiplier = 1,activation='relu')) #28
model1.add(Dropout(0.15))
model1.add(BatchNormalization())
# 28*28*64
#Jin=1,s=1, RFi=3, k=3, Jout=1
#receptive field =3+(3-1)*1=5
model1.add(SeparableConv2D(filters= 128,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay),depth_multiplier = 1,activation='relu')) #26
model1.add(BatchNormalization())
model1.add(Dropout(0.15))
# 26*26*128
#Jin=1,s=1, RFi=5, k=3, Jout=1
#receptive field =5+(3-1)*1=7
model1.add(MaxPooling2D(pool_size=(2, 2))) #13
# 13*13*32
#Jin=1,s=2, RFi=7, k=2, Jout=2
#receptive field =7+(2-1)*1=8
#receptive field =8

model1.add(SeparableConv2D(filters= 64,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay),depth_multiplier = 1,activation='relu')) #11
model1.add(BatchNormalization())
model1.add(Dropout(0.15))
# 11*11*64
#Jin=2,s=1, RFi=8, k=3, Jout=2
#receptive field =8+(3-1)*2=12
#receptive field =12

model1.add(SeparableConv2D(filters= 128,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay),depth_multiplier = 1,activation='relu')) #9
model1.add(BatchNormalization())
model1.add(Dropout(0.15))
# 9*9*128
#Jin=2,s=1, RFi=12, k=3, Jout=2
#receptive field =12+(3-1)*2=16
#receptive field =16


model1.add(SeparableConv2D(filters= 256,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay),depth_multiplier = 1,activation='relu')) #7
model1.add(BatchNormalization())
model1.add(Dropout(0.15))
# 7*7*256
#Jin=2,s=1, RFi=16, k=3, Jout=2
#receptive field =16+(3-1)*2=20
#receptive field =20


model1.add(MaxPooling2D(pool_size=(2, 2))) 

# 3*3*256
#Jin=2,s=2, RFi=20, k=2, Jout=4
#receptive field =20+(2-1)*2=22
#receptive field =22

model1.add(SeparableConv2D(filters= 32,kernel_size=(1,1),kernel_regularizer=regularizers.l2(weight_decay),depth_multiplier = 1,activation='relu')) 
model1.add(BatchNormalization())
model1.add(Dropout(0.15))

# 3*3*32
#Jin=4,s=1, RFi=22, k=1, Jout=4
#receptive field =22+(1-1)*4=22
#receptive field =22
model1.add(SeparableConv2D(filters= 10,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay),depth_multiplier = 1,activation='relu')) 

# 1*1*10
#Jin=4,s=1, RFi=22, k=3, Jout=4
#receptive field =22+(3-1)*4=30
#receptive field =30
model1.add(GlobalAveragePooling2D())
model1.add(Activation('softmax'))

model1.summary()
------------------------------------------------------------------------------------------------------------------------
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
def scheduler(epoch, lr):
  return round(0.005 * 1/(1 + 0.319 * epoch), 10)

model1.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.003), metrics=['accuracy'])



from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(zoom_range=0.0, 
                             horizontal_flip=False)


# train the model1
start = time.time()
# Train the model1
model1_info = model1.fit_generator(datagen.flow(train_features, train_labels, batch_size = 100 ),
                                 samples_per_epoch = train_features.shape[0], nb_epoch = 50, 
                                 validation_data = (test_features, test_labels), callbacks=[LearningRateScheduler(scheduler, verbose=1)])
end = time.time()
print ("Model took %0.2f seconds to train"%(end - start))
# plot model1 history
plot_model_history(model1_info)
# compute test accuracy
print ("Accuracy on test data is: %0.2f"%accuracy(test_features, test_labels, model1))

-----------------------------------------------------------------------------------------------------------------------------------------------
3. Log of 50 epochs
Epoch 00040: LearningRateScheduler setting learning rate to 0.0003719961.
500/500 [==============================] - 28s 55ms/step - loss: 0.2988 - acc: 0.8916 - val_loss: 0.6966 - val_acc: 0.7842
Epoch 41/50

Epoch 00041: LearningRateScheduler setting learning rate to 0.0003633721.
500/500 [==============================] - 28s 55ms/step - loss: 0.2973 - acc: 0.8934 - val_loss: 0.6747 - val_acc: 0.7889
Epoch 42/50

Epoch 00042: LearningRateScheduler setting learning rate to 0.0003551389.
500/500 [==============================] - 28s 55ms/step - loss: 0.3020 - acc: 0.8924 - val_loss: 0.6984 - val_acc: 0.7863
Epoch 43/50

Epoch 00043: LearningRateScheduler setting learning rate to 0.0003472705.
500/500 [==============================] - 28s 55ms/step - loss: 0.2950 - acc: 0.8933 - val_loss: 0.6650 - val_acc: 0.7896
Epoch 44/50

Epoch 00044: LearningRateScheduler setting learning rate to 0.0003397432.
500/500 [==============================] - 28s 55ms/step - loss: 0.2933 - acc: 0.8941 - val_loss: 0.6738 - val_acc: 0.7930
Epoch 45/50

Epoch 00045: LearningRateScheduler setting learning rate to 0.0003325352.
500/500 [==============================] - 28s 55ms/step - loss: 0.2939 - acc: 0.8955 - val_loss: 0.6557 - val_acc: 0.7924
Epoch 46/50

Epoch 00046: LearningRateScheduler setting learning rate to 0.0003256268.
500/500 [==============================] - 28s 55ms/step - loss: 0.2956 - acc: 0.8938 - val_loss: 0.6419 - val_acc: 0.7995
Epoch 47/50

Epoch 00047: LearningRateScheduler setting learning rate to 0.0003189996.
500/500 [==============================] - 28s 55ms/step - loss: 0.2908 - acc: 0.8949 - val_loss: 0.6694 - val_acc: 0.7930
Epoch 48/50

Epoch 00048: LearningRateScheduler setting learning rate to 0.0003126368.
500/500 [==============================] - 28s 55ms/step - loss: 0.2887 - acc: 0.8966 - val_loss: 0.6700 - val_acc: 0.7921
Epoch 49/50

Epoch 00049: LearningRateScheduler setting learning rate to 0.0003065228.
500/500 [==============================] - 27s 55ms/step - loss: 0.2894 - acc: 0.8966 - val_loss: 0.6760 - val_acc: 0.7901
Epoch 50/50

Epoch 00050: LearningRateScheduler setting learning rate to 0.0003006434.
500/500 [==============================] - 28s 55ms/step - loss: 0.2857 - acc: 0.8973 - val_loss: 0.6628 - val_acc: 0.7950
Model took 1376.25 seconds to train

Accuracy on test data is: 79.50
