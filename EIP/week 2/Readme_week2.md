Best model with highest validation accuracy of 99.45 at 19th Epoch 

1.Log for 20 Epochs 
  Train on 60000 samples, validate on 10000 samples
  Epoch 1/20

  Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
  60000/60000 [==============================] - 19s 313us/step - loss: 0.0239 - acc: 0.9924 - val_loss: 0.0279 - val_acc: 0.9916
  Epoch 2/20

  Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.
  60000/60000 [==============================] - 7s 118us/step - loss: 0.0184 - acc: 0.9937 - val_loss: 0.0267 - val_acc: 0.9918
  Epoch 3/20

  Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.
  60000/60000 [==============================] - 7s 115us/step - loss: 0.0129 - acc: 0.9956 - val_loss: 0.0239 - val_acc: 0.9920
  Epoch 4/20

  Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.
  60000/60000 [==============================] - 7s 114us/step - loss: 0.0108 - acc: 0.9968 - val_loss: 0.0276 - val_acc: 0.9917
  Epoch 5/20

  Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
  60000/60000 [==============================] - 7s 113us/step - loss: 0.0087 - acc: 0.9972 - val_loss: 0.0278 - val_acc: 0.9929
  Epoch 6/20

  Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.
  60000/60000 [==============================] - 7s 115us/step - loss: 0.0088 - acc: 0.9972 - val_loss: 0.0267 - val_acc: 0.9923
  Epoch 7/20

  Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
  60000/60000 [==============================] - 7s 113us/step - loss: 0.0092 - acc: 0.9968 - val_loss: 0.0269 - val_acc: 0.9926
  Epoch 8/20

  Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
  60000/60000 [==============================] - 7s 113us/step - loss: 0.0075 - acc: 0.9975 - val_loss: 0.0259 - val_acc: 0.9928
  Epoch 9/20

  Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
  60000/60000 [==============================] - 7s 113us/step - loss: 0.0067 - acc: 0.9978 - val_loss: 0.0256 - val_acc: 0.9931
  Epoch 10/20

  Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
  60000/60000 [==============================] - 7s 113us/step - loss: 0.0058 - acc: 0.9981 - val_loss: 0.0247 - val_acc: 0.9933
  Epoch 11/20

  Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.
  60000/60000 [==============================] - 7s 112us/step - loss: 0.0056 - acc: 0.9981 - val_loss: 0.0254 - val_acc: 0.9936
  Epoch 12/20

  Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.
  60000/60000 [==============================] - 7s 113us/step - loss: 0.0058 - acc: 0.9984 - val_loss: 0.0252 - val_acc: 0.9941
  Epoch 13/20

  Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.
  60000/60000 [==============================] - 7s 113us/step - loss: 0.0048 - acc: 0.9985 - val_loss: 0.0286 - val_acc: 0.9932
  Epoch 14/20

  Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.
  60000/60000 [==============================] - 7s 113us/step - loss: 0.0053 - acc: 0.9982 - val_loss: 0.0251 - val_acc: 0.9929
  Epoch 15/20

  Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.
  60000/60000 [==============================] - 7s 114us/step - loss: 0.0040 - acc: 0.9988 - val_loss: 0.0281 - val_acc: 0.9933
  Epoch 16/20

  Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.
  60000/60000 [==============================] - 7s 114us/step - loss: 0.0045 - acc: 0.9987 - val_loss: 0.0284 - val_acc: 0.9940
  Epoch 17/20

  Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
  60000/60000 [==============================] - 7s 116us/step - loss: 0.0042 - acc: 0.9987 - val_loss: 0.0280 - val_acc: 0.9934
  Epoch 18/20

  Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
  60000/60000 [==============================] - 7s 114us/step - loss: 0.0041 - acc: 0.9988 - val_loss: 0.0282 - val_acc: 0.9936
  Epoch 19/20

  Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
  60000/60000 [==============================] - 7s 114us/step - loss: 0.0036 - acc: 0.9988 - val_loss: 0.0263 - val_acc: 0.9945
  Epoch 20/20

  Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
  60000/60000 [==============================] - 7s 113us/step - loss: 0.0031 - acc: 0.9991 - val_loss: 0.0263 - val_acc: 0.9933
  <keras.call5 usebacks.History at 0x7fe99d441cc0>



2. model.evaluate
  [0.026320638077960756, 0.9933]
  

3. Startegy used
two convolution blocks used. batchnormalisation is used after all the layers except the last layer to amplify the features extracted.
1st block - 3 convolution layers with filters 10, 16, 32 used.
dropout of 20% used to avoid overfitting
2nd convolution block- 3 convolution layers with filters 10, 16, 16 used.
dropout of 20% used to avoid overfitting

