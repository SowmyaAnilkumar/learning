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


3. Log of 50 epochs
Epoch 1/50

Epoch 00001: LearningRateScheduler setting learning rate to 0.005.
390/390 [==============================] - 33s 85ms/step - loss: 1.4791 - acc: 0.4675 - val_loss: 1.4573 - val_acc: 0.5402
Epoch 2/50

Epoch 00002: LearningRateScheduler setting learning rate to 0.0037907506.
390/390 [==============================] - 27s 69ms/step - loss: 1.0733 - acc: 0.6199 - val_loss: 1.0831 - val_acc: 0.6203
Epoch 3/50

Epoch 00003: LearningRateScheduler setting learning rate to 0.0030525031.
390/390 [==============================] - 27s 69ms/step - loss: 0.9346 - acc: 0.6718 - val_loss: 1.0347 - val_acc: 0.6399
Epoch 4/50

Epoch 00004: LearningRateScheduler setting learning rate to 0.002554931.
390/390 [==============================] - 27s 69ms/step - loss: 0.8433 - acc: 0.7031 - val_loss: 0.8530 - val_acc: 0.7048
Epoch 5/50

Epoch 00005: LearningRateScheduler setting learning rate to 0.0021968366.
390/390 [==============================] - 27s 69ms/step - loss: 0.7768 - acc: 0.7298 - val_loss: 0.9117 - val_acc: 0.6921
Epoch 6/50

Epoch 00006: LearningRateScheduler setting learning rate to 0.0019267823.
390/390 [==============================] - 27s 69ms/step - loss: 0.7308 - acc: 0.7446 - val_loss: 0.8438 - val_acc: 0.7063
Epoch 7/50

Epoch 00007: LearningRateScheduler setting learning rate to 0.0017158545.
390/390 [==============================] - 27s 69ms/step - loss: 0.6906 - acc: 0.7568 - val_loss: 0.8226 - val_acc: 0.7119
Epoch 8/50

Epoch 00008: LearningRateScheduler setting learning rate to 0.0015465512.
390/390 [==============================] - 27s 69ms/step - loss: 0.6652 - acc: 0.7669 - val_loss: 0.7898 - val_acc: 0.7204
Epoch 9/50

Epoch 00009: LearningRateScheduler setting learning rate to 0.0014076577.
390/390 [==============================] - 27s 69ms/step - loss: 0.6416 - acc: 0.7750 - val_loss: 0.8046 - val_acc: 0.7208
Epoch 10/50

Epoch 00010: LearningRateScheduler setting learning rate to 0.0012916559.
390/390 [==============================] - 27s 69ms/step - loss: 0.6153 - acc: 0.7848 - val_loss: 0.7999 - val_acc: 0.7264
Epoch 11/50

Epoch 00011: LearningRateScheduler setting learning rate to 0.0011933174.
390/390 [==============================] - 27s 69ms/step - loss: 0.6017 - acc: 0.7870 - val_loss: 0.7106 - val_acc: 0.7519
Epoch 12/50

Epoch 00012: LearningRateScheduler setting learning rate to 0.0011088933.
390/390 [==============================] - 27s 69ms/step - loss: 0.5822 - acc: 0.7964 - val_loss: 0.8145 - val_acc: 0.7275
Epoch 13/50

Epoch 00013: LearningRateScheduler setting learning rate to 0.0010356255.
390/390 [==============================] - 27s 69ms/step - loss: 0.5685 - acc: 0.8001 - val_loss: 0.7052 - val_acc: 0.7571
Epoch 14/50

Epoch 00014: LearningRateScheduler setting learning rate to 0.0009714397.
390/390 [==============================] - 27s 69ms/step - loss: 0.5561 - acc: 0.8054 - val_loss: 0.6652 - val_acc: 0.7702
Epoch 15/50

Epoch 00015: LearningRateScheduler setting learning rate to 0.0009147457.
390/390 [==============================] - 27s 69ms/step - loss: 0.5439 - acc: 0.8075 - val_loss: 0.6826 - val_acc: 0.7631
Epoch 16/50

Epoch 00016: LearningRateScheduler setting learning rate to 0.0008643042.
390/390 [==============================] - 27s 69ms/step - loss: 0.5328 - acc: 0.8144 - val_loss: 0.6794 - val_acc: 0.7707
Epoch 17/50

Epoch 00017: LearningRateScheduler setting learning rate to 0.000819135.
390/390 [==============================] - 27s 69ms/step - loss: 0.5227 - acc: 0.8150 - val_loss: 0.6828 - val_acc: 0.7677
Epoch 18/50

Epoch 00018: LearningRateScheduler setting learning rate to 0.0007784524.
390/390 [==============================] - 27s 69ms/step - loss: 0.5130 - acc: 0.8185 - val_loss: 0.6997 - val_acc: 0.7627
Epoch 19/50

Epoch 00019: LearningRateScheduler setting learning rate to 0.0007416197.
390/390 [==============================] - 27s 70ms/step - loss: 0.5059 - acc: 0.8210 - val_loss: 0.6560 - val_acc: 0.7775
Epoch 20/50

Epoch 00020: LearningRateScheduler setting learning rate to 0.000708115.
390/390 [==============================] - 27s 69ms/step - loss: 0.4947 - acc: 0.8233 - val_loss: 0.7389 - val_acc: 0.7504
Epoch 21/50

Epoch 00021: LearningRateScheduler setting learning rate to 0.0006775068.
390/390 [==============================] - 27s 69ms/step - loss: 0.4882 - acc: 0.8285 - val_loss: 0.6763 - val_acc: 0.7720
Epoch 22/50

Epoch 00022: LearningRateScheduler setting learning rate to 0.000649435.
390/390 [==============================] - 27s 69ms/step - loss: 0.4846 - acc: 0.8284 - val_loss: 0.6401 - val_acc: 0.7853
Epoch 23/50

Epoch 00023: LearningRateScheduler setting learning rate to 0.0006235969.
390/390 [==============================] - 27s 69ms/step - loss: 0.4719 - acc: 0.8323 - val_loss: 0.6591 - val_acc: 0.7804
Epoch 24/50

Epoch 00024: LearningRateScheduler setting learning rate to 0.0005997361.
390/390 [==============================] - 27s 69ms/step - loss: 0.4704 - acc: 0.8345 - val_loss: 0.6741 - val_acc: 0.7700
Epoch 25/50

Epoch 00025: LearningRateScheduler setting learning rate to 0.000577634.
390/390 [==============================] - 27s 69ms/step - loss: 0.4597 - acc: 0.8365 - val_loss: 0.6494 - val_acc: 0.7819
Epoch 26/50

Epoch 00026: LearningRateScheduler setting learning rate to 0.0005571031.
390/390 [==============================] - 27s 69ms/step - loss: 0.4592 - acc: 0.8387 - val_loss: 0.7056 - val_acc: 0.7632
Epoch 27/50

Epoch 00027: LearningRateScheduler setting learning rate to 0.0005379815.
390/390 [==============================] - 27s 69ms/step - loss: 0.4581 - acc: 0.8385 - val_loss: 0.6550 - val_acc: 0.7808
Epoch 28/50

Epoch 00028: LearningRateScheduler setting learning rate to 0.000520129.
390/390 [==============================] - 27s 69ms/step - loss: 0.4509 - acc: 0.8397 - val_loss: 0.6546 - val_acc: 0.7811
Epoch 29/50

Epoch 00029: LearningRateScheduler setting learning rate to 0.0005034233.
390/390 [==============================] - 27s 69ms/step - loss: 0.4437 - acc: 0.8443 - val_loss: 0.6191 - val_acc: 0.7887
Epoch 30/50

Epoch 00030: LearningRateScheduler setting learning rate to 0.0004877573.
390/390 [==============================] - 27s 69ms/step - loss: 0.4388 - acc: 0.8445 - val_loss: 0.6566 - val_acc: 0.7771
Epoch 31/50

Epoch 00031: LearningRateScheduler setting learning rate to 0.0004730369.
390/390 [==============================] - 27s 69ms/step - loss: 0.4361 - acc: 0.8459 - val_loss: 0.6405 - val_acc: 0.7826
Epoch 32/50

Epoch 00032: LearningRateScheduler setting learning rate to 0.000459179.
390/390 [==============================] - 27s 69ms/step - loss: 0.4272 - acc: 0.8488 - val_loss: 0.6726 - val_acc: 0.7738
Epoch 33/50

Epoch 00033: LearningRateScheduler setting learning rate to 0.0004461099.
390/390 [==============================] - 27s 69ms/step - loss: 0.4276 - acc: 0.8480 - val_loss: 0.6549 - val_acc: 0.7793
Epoch 34/50

Epoch 00034: LearningRateScheduler setting learning rate to 0.0004337642.
390/390 [==============================] - 27s 69ms/step - loss: 0.4199 - acc: 0.8510 - val_loss: 0.6474 - val_acc: 0.7812
Epoch 35/50

Epoch 00035: LearningRateScheduler setting learning rate to 0.0004220834.
390/390 [==============================] - 27s 69ms/step - loss: 0.4259 - acc: 0.8495 - val_loss: 0.6808 - val_acc: 0.7721
Epoch 36/50

Epoch 00036: LearningRateScheduler setting learning rate to 0.0004110152.
390/390 [==============================] - 27s 69ms/step - loss: 0.4154 - acc: 0.8518 - val_loss: 0.6795 - val_acc: 0.7720
Epoch 37/50

Epoch 00037: LearningRateScheduler setting learning rate to 0.0004005127.
390/390 [==============================] - 27s 69ms/step - loss: 0.4140 - acc: 0.8530 - val_loss: 0.6717 - val_acc: 0.7760
Epoch 38/50

Epoch 00038: LearningRateScheduler setting learning rate to 0.0003905335.
390/390 [==============================] - 27s 69ms/step - loss: 0.4142 - acc: 0.8530 - val_loss: 0.6550 - val_acc: 0.7814
Epoch 39/50

Epoch 00039: LearningRateScheduler setting learning rate to 0.0003810395.
390/390 [==============================] - 27s 69ms/step - loss: 0.4081 - acc: 0.8562 - val_loss: 0.6735 - val_acc: 0.7740
Epoch 40/50

Epoch 00040: LearningRateScheduler setting learning rate to 0.0003719961.
390/390 [==============================] - 27s 69ms/step - loss: 0.4059 - acc: 0.8544 - val_loss: 0.6834 - val_acc: 0.7739
Epoch 41/50

Epoch 00041: LearningRateScheduler setting learning rate to 0.0003633721.
390/390 [==============================] - 27s 69ms/step - loss: 0.4089 - acc: 0.8543 - val_loss: 0.6505 - val_acc: 0.7843
Epoch 42/50

Epoch 00042: LearningRateScheduler setting learning rate to 0.0003551389.
390/390 [==============================] - 27s 70ms/step - loss: 0.3976 - acc: 0.8594 - val_loss: 0.6430 - val_acc: 0.7851
Epoch 43/50

Epoch 00043: LearningRateScheduler setting learning rate to 0.0003472705.
390/390 [==============================] - 27s 69ms/step - loss: 0.3952 - acc: 0.8595 - val_loss: 0.6678 - val_acc: 0.7773
Epoch 44/50

Epoch 00044: LearningRateScheduler setting learning rate to 0.0003397432.
390/390 [==============================] - 27s 69ms/step - loss: 0.3939 - acc: 0.8602 - val_loss: 0.6478 - val_acc: 0.7846
Epoch 45/50

Epoch 00045: LearningRateScheduler setting learning rate to 0.0003325352.
390/390 [==============================] - 27s 69ms/step - loss: 0.3961 - acc: 0.8589 - val_loss: 0.6486 - val_acc: 0.7852
Epoch 46/50

Epoch 00046: LearningRateScheduler setting learning rate to 0.0003256268.
390/390 [==============================] - 27s 69ms/step - loss: 0.3898 - acc: 0.8592 - val_loss: 0.6586 - val_acc: 0.7829
Epoch 47/50

Epoch 00047: LearningRateScheduler setting learning rate to 0.0003189996.
390/390 [==============================] - 27s 69ms/step - loss: 0.3876 - acc: 0.8626 - val_loss: 0.6655 - val_acc: 0.7805
Epoch 48/50

Epoch 00048: LearningRateScheduler setting learning rate to 0.0003126368.
390/390 [==============================] - 27s 69ms/step - loss: 0.3880 - acc: 0.8617 - val_loss: 0.6459 - val_acc: 0.7895
Epoch 49/50

Epoch 00049: LearningRateScheduler setting learning rate to 0.0003065228.
390/390 [==============================] - 27s 69ms/step - loss: 0.3793 - acc: 0.8652 - val_loss: 0.6348 - val_acc: 0.7899
Epoch 50/50

Epoch 00050: LearningRateScheduler setting learning rate to 0.0003006434.
390/390 [==============================] - 27s 69ms/step - loss: 0.3861 - acc: 0.8628 - val_loss: 0.6495 - val_acc: 0.7850
Model took 1355.69 seconds to train

Accuracy on test data is: 78.50