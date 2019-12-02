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

model1.add(SeparableConv2D(filters= 128,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay),depth_multiplier = 1,activation='relu')) #11
model1.add(BatchNormalization())
model1.add(Dropout(0.15))

# 11*11*128
#Jin=2,s=1, RFi=8, k=3, Jout=2
#receptive field =8+(3-1)*2=12
#receptive field =12

model1.add(SeparableConv2D(filters= 256,kernel_size=(3,3),kernel_regularizer=regularizers.l2(weight_decay),depth_multiplier = 1,activation='relu')) #9
model1.add(BatchNormalization())
model1.add(Dropout(0.15))

# 9*9*256
#Jin=2,s=1, RFi=12, k=3, Jout=2
#receptive field =12+(3-1)*2=16
#receptive field =16

model1.add(MaxPooling2D(pool_size=(2, 2))) 

# 4*4*256
#Jin=2,s=2, RFi=16, k=2, Jout=4
#receptive field =16+(2-1)*2=18
#receptive field =18

model1.add(SeparableConv2D(filters= 32,kernel_size=(1,1),kernel_regularizer=regularizers.l2(weight_decay),depth_multiplier = 1,activation='relu')) 
model1.add(BatchNormalization())
model1.add(Dropout(0.15))

# 4*4*32
#Jin=4,s=1, RFi=18, k=1, Jout=4
#receptive field =18+(1-1)*4=18
#receptive field =18
model1.add(SeparableConv2D(filters= 10,kernel_size=(4,4),kernel_regularizer=regularizers.l2(weight_decay),depth_multiplier = 1,activation='relu')) 

# 1*1*10
#Jin=4,s=1, RFi=18, k=4, Jout=4
#receptive field =18+(4-1)*4=18
#receptive field =30
model1.add(GlobalAveragePooling2D())
model1.add(Activation('softmax'))

model1.summary()


3. Log of 50 epochs

