import keras
from keras import layers, models
from keras.layers.merge import concatenate, add, subtract
import keras.backend as K
import numpy as np
class CapsuleNet_Model():
    def __init__(self, input_size=(512,512), number_of_class=3):
        self.input_size = input_size
        self.number_of_class = number_of_class

class DualModel():
    def Conv2DBNSLU(self, x, filters, kernel_size=1, strides=1, padding='same', activation="relu", name="", bias=False, bn=True, scale=False):
        x = layers.Conv2D(
            filters,
            kernel_size = kernel_size,
            strides=strides,
            padding=padding,
            name=name,
            use_bias=bias)(x)
        if bn:
            x = layers.BatchNormalization(scale=scale)(x)
        x = layers.Activation(activation)(x)
        return x
    def __init__(self, input_shape=(256,256), number_of_class=3,number_of_channel=1):
        self.input_shape = input_shape
        self.number_of_class = number_of_class
        self.number_of_channel= number_of_channel

    def CoreModelStructure(self, init_filter=32, reshape=False, activation_last="softmax"):
        n_class = self.number_of_class
        number_of_channel = self.number_of_channel
        shape_default = (self.input_shape[0],self.input_shape[1],number_of_channel)
        x = layers.Input(shape=shape_default)
        y_predict = self.CoreModel(x, "ucent", init_filter, n_class, "ucent", reshape, 1,  activation_last=activation_last)
        train_model = models.Model(x, y_predict )
        return train_model, train_model
    
    #Root of this code, Here begin the call
    def CoreModelDual(self, init_filter=32, reshape=False, activation_last="softmax"):
        n_class = self.number_of_class
        number_of_channel = self.number_of_channel
        shape_default = (self.input_shape[0],self.input_shape[1],number_of_channel)
        x = layers.Input(shape=shape_default)
        y_predict = self.CoreModel(x, "ucent", init_filter, n_class, "ucent", reshape, 1,  activation_last=activation_last)
        bol = layers.concatenate([y_predict, x])
        seg = self.SegModel(bol, "predict_to_seg_02", 1, "sigmoid", 1)
        output = layers.concatenate([y_predict, seg, x])
        contour = self.SegModel_Contour(
            output, "predict_to_contour_01", 1, "sigmoid", 1)
        train_model = models.Model(x, [y_predict,seg, contour])
        eval_model = models.Model(x, [y_predict,seg, contour])
        return train_model, eval_model

    def Inception_V3(self, final_activation="softmax"):
        from keras.applications.inception_v3 import InceptionV3
        import keras.backend as K
        import numpy as np
        shape_default = (self.input_shape[0], self.input_shape[1], self.number_of_channel)
        x = layers.Input(shape=shape_default)
        
        Inception_V3 = InceptionV3(input_shape=shape_default, weights='imagenet', include_top=False)

        y = Inception_V3.output
        y = layers.GlobalAveragePooling2D()(y)
        y = layers.Dense(2048, activation='relu')(y)
        y = layers.Dropout(0.5)(y)
        y = layers.Dense(1024, activation='relu')(y)
        y = layers.Dropout(0.5)(y)
        y = layers.Lambda(lambda  x_: K.l2_normalize(y,axis=1))(y)

        shape_default = (shape_default[0], shape_default[1], self.number_of_class)
        y = layers.Dense(np.prod(shape_default), activation=final_activation)(y)
        y = layers.Reshape(target_shape=shape_default, name='out_recon')(y)
        convnet_model = models.Model(inputs=Inception_V3.input, outputs=y)
        return convnet_model,convnet_model

    def CoreModel(self,x, class_name, n_filter=32,n_class=3, output_name="", reshape=True, kernel_size_softmax=1, activation_last="softmax", factor_kernel=1):
        
        #512x512 Level 3 --> 512x512x32
        conv_level_5 = self.Conv2DBNSLU(x=x, filters=n_filter, kernel_size=(3*factor_kernel,3*factor_kernel), strides=1,
                                padding='same', activation='selu', name='%s_conv_level_512x512x32'% (output_name)) #5,4
        
        #512x512x32 --> 256x256x64
        conv_level_4 = layers.Conv2D(filters=int(round(n_filter*2)), kernel_size=3*factor_kernel, strides=1,
                                padding='same', activation='selu', name= '%s_conv_level_256x256x64' %(output_name))(conv_level_5)
        conv_level_4 = layers.MaxPooling2D(pool_size=(2,2),strides=2)(conv_level_4) #5

        #256x256x64 --> 64x64x96
        conv_level_3 = layers.Conv2D(filters=int(round(n_filter*4)), kernel_size=3*factor_kernel, strides=1,
                            padding='same', activation='selu', name= '%s_conv_level_128x128x96' %(output_name))(conv_level_4)
        conv_level_3 = layers.MaxPooling2D(pool_size=(2,2),strides=2)(conv_level_3) #4
        
        #64x64x96 --> 16x16x128
        conv_level_2 = layers.Conv2D(filters=int(round(n_filter*8)), kernel_size=3*factor_kernel, strides=2,
                            padding='same', activation='selu', name= '%s_conv_level_16x16x128' %(output_name))(conv_level_3)
        conv_level_2 = layers.MaxPooling2D(pool_size=(2,2),strides=2)(conv_level_2) #4
        
        #16x16x128 --> 8x8x256
        conv_level_1 = layers.Conv2D(filters=int(round(n_filter*16)), kernel_size=3*factor_kernel, strides=1,
                            padding='same', activation='selu', name= '%s_conv_level_8x8x256' %(output_name))(conv_level_2)
        conv_level_1 = layers.MaxPooling2D(pool_size=(2,2),strides=2)(conv_level_1)

        #16x16 level 1 --> 8x8 (4,5)
        conv_level_1 = self.Conv2DBNSLU(x=conv_level_2, filters=int(round(n_filter*24)), kernel_size=(3*factor_kernel,3*factor_kernel), strides=1,
                                padding='same', activation='selu', name='%s_conv_level_1_0'% (output_name))
        conv_level_1 = layers.MaxPooling2D(pool_size=(2,2),strides=2)(conv_level_1)
        conv_level_1 = layers.SpatialDropout2D(0.2)(conv_level_1)

        #subgroups 
        conv_level_1_7 = layers.Conv2D(filters=int(round(n_filter*4)), kernel_size=7, strides=1, padding='same', activation='selu', name='%s_conv_level_1_7x7' %(output_name)) (conv_level_1)
        conv_level_1_7 = layers.Conv2D(filters=int(round(n_filter*4)), kernel_size=(1,7), strides=1, padding='same', activation='selu', name='%s_conv_level_1_1x7' %(output_name)) (conv_level_1_7)
        conv_level_1_7 = layers.Conv2D(filters=int(round(n_filter*4)), kernel_size=(7,1), strides=1, padding='same', activation='selu', name='%s_conv_level_1_7x1'% (output_name)) (conv_level_1_7)
        conv_level_1_7 = layers.SpatialDropout2D(0.2)(conv_level_1_7)
        
        conv_level_1_5 = layers.Conv2D(filters=int(round(n_filter*4)), kernel_size=5, strides=1, padding='same', activation='selu', name='%s_conv_level_1_0_5x5' % (output_name)) (conv_level_1)
        conv_level_1_5 = layers.Conv2D(filters=int(round(n_filter*4)), kernel_size=(5,1*factor_kernel), strides=1, padding='same', activation='selu', name='%s_conv_level_1_5x1' % (output_name)) (conv_level_1_5)
        conv_level_1_5 = layers.Conv2D(filters=int(round(n_filter*4)), kernel_size=(1*factor_kernel,5), strides=1, padding='same', activation='selu', name='%s_conv_level_1_1x5' % (output_name )) (conv_level_1_5)
        conv_level_1_5 = layers.SpatialDropout2D(0.2)(conv_level_1_5)

        conv_level_1_3 = layers.Conv2D(filters=int(round(n_filter*4)), kernel_size=3, strides=1, padding='same', activation='selu', name='%s_conv_level_1_3' %(output_name))(conv_level_1)
        conv_level_1_3 = layers.Conv2D(filters=int(round(n_filter*4)), kernel_size=(3,1*factor_kernel), strides=1, padding='same', activation='selu', name='%s_conv_level_1_1_3x1' % (output_name)) (conv_level_1_3)
        conv_level_1_3 = layers.Conv2D(filters=int(round(n_filter*4)), kernel_size=(1*factor_kernel, 3), strides=1,
                                       padding='same', activation='selu', name='%s_conv_level_1_2_1x3' % (output_name))(conv_level_1_3)
        conv_level_1_3 = layers.SpatialDropout2D(0.2)(conv_level_1_3)
        #Concatenate
        conv_level_1 = concatenate([conv_level_1,
                                    conv_level_1_5,
                                    conv_level_1_3,
                                    conv_level_1_7
                                    ])
        
        #8 -> 16
        upsample_level_0_to_1 = keras.layers.UpSampling2D()(conv_level_1)
        upsample_level_0_to_1 = self.Conv2DBNSLU(x=upsample_level_0_to_1, filters=int(round(n_filter*16)), kernel_size=(3*factor_kernel,3*factor_kernel), strides=1,
                                padding='same', activation='selu', name='%s_conv_level_16x16x_Up'% (output_name))
        conv_level_2 = layers.Conv2D(filters=int(round(n_filter*16)), kernel_size=3*factor_kernel, strides=1,
                            padding='same', activation='selu', name= '%s_conv_level_16x16xA_Up' %(output_name))(conv_level_2) #4
        conv_level_0_to_1 = concatenate([upsample_level_0_to_1, conv_level_2])
        conv_level_0_to_1 = layers.Conv2D(filters=int(round(n_filter*16)), kernel_size=3*factor_kernel, strides=1,
                            padding='same', activation='selu', name= '%s_conv_level_16x16xB_Up' %(output_name))(conv_level_0_to_1) #4
        
        #16 -> 64
        upsample_level_1_to_2 = keras.layers.UpSampling2D((4,4))(conv_level_0_to_1)
        upsample_level_1_to_2 = self.Conv2DBNSLU(x=upsample_level_1_to_2, filters=int(round(n_filter*8)), kernel_size=(3*factor_kernel,3*factor_kernel), strides=1,
                                padding='same', activation='selu', name='%s_conv_level_64x64x_Up'% (output_name)) #4
        conv_level_3 = layers.Conv2D(filters=int(round(n_filter*8)), kernel_size=3*factor_kernel, strides=1,
                            padding='same', activation='selu', name= '%s_conv_level_64x64xA_Up' %(output_name))(conv_level_3) #4
        conv_level_1_to_2 = concatenate([upsample_level_1_to_2, conv_level_3])
        conv_level_1_to_2 = layers.Conv2D(filters=int(round(n_filter*8)), kernel_size=3*factor_kernel, strides=1,
                            padding='same', activation='selu', name= '%s_conv_level_64x64xB_Up' %(output_name))(conv_level_1_to_2) #4
        
        ##64 -> 256
        upsample_level_2_to_3 = keras.layers.UpSampling2D((2,2))(conv_level_1_to_2)
        upsample_level_2_to_3 = self.Conv2DBNSLU(x=upsample_level_2_to_3, filters=int(round(n_filter*4)), kernel_size=(3*factor_kernel,3*factor_kernel), strides=1,
                                padding='same', activation='selu', name='%s_conv_level_256x256x_Up'% (output_name)) #3,3
        conv_level_4 = layers.Conv2D(filters=int(round(n_filter*3)), kernel_size=3*factor_kernel, strides=1,
                            padding='same', activation='selu', name= '%s_conv_level_256x256xA_Up' %(output_name))(conv_level_4) #4
        conv_level_2_to_3 = concatenate([upsample_level_2_to_3, conv_level_4])
        conv_level_2_to_3 = layers.Conv2D(filters=int(round(n_filter*2)), kernel_size=3*factor_kernel, strides=1,
                            padding='same', activation='selu', name= '%s_conv_level_256x256xB_Up' %(output_name))(conv_level_2_to_3)#4
        
        ##256 -> 512
        upsample_level_3_to_4 = keras.layers.UpSampling2D()(conv_level_2_to_3)
        upsample_level_3_to_4 = self.Conv2DBNSLU(x=upsample_level_3_to_4, filters=int(round(n_filter)), kernel_size=(3*factor_kernel,4*factor_kernel), strides=1,
                                padding='same', activation='selu', name='%s_conv_level_512x512x_Up'% (output_name)) #3,4
        conv_level_3_to_4 = concatenate([upsample_level_3_to_4, conv_level_5])
        conv_level_3_to_4 = layers.Conv2D(filters=int(round(n_filter)), kernel_size=3*factor_kernel, strides=1,
                            padding='same', activation='selu', name= '%s_conv_level_512x512xB_Up' %(output_name))(conv_level_3_to_4) #4
        

        #Softmax
        one_hot_layer_1 = layers.Conv2D(filters=n_class, kernel_size=1, strides=1,
                                        padding='same', activation=activation_last, name='%ss_ucnet11'%(output_name))(conv_level_3_to_4) #name='%s_ucnet01' % (output_name))(conv_level_3_to_4)
            
        one_hot_layer_2 = layers.Conv2D(filters=n_class, kernel_size=(5*factor_kernel,5*factor_kernel), strides=1,
                                    padding='same', activation=activation_last, name='%s_ucnet55'%(output_name))(conv_level_3_to_4)                
        
        one_hot_layer_3 = layers.Conv2D(filters=n_class, kernel_size=(3*factor_kernel,3*factor_kernel), strides=1,
                                    padding='same', activation=activation_last, name='%s_ucnet_33'%(output_name))(conv_level_3_to_4)
        #one_hot_layer_3 = layers.UpSampling2D(size=(4,4))(one_hot_layer_3)
        #Average layer
        one_hot_layer = keras.layers.average(
            [one_hot_layer_2, one_hot_layer_3, one_hot_layer_1], name=output_name)
        
        '''
        one_hot_layer = layers.Conv2D(filters=n_class, kernel_size=1, strides=1, padding='same',
                                        activation=activation_last, name=output_name)(conv_level_3_to_4)
        one_hot_layer = layers.Conv2D(filters=n_class, kernel_size=1, strides=1, padding='same',
                                        activation=activation_last, name=output_name)(conv_level_3_to_4)
        one_hot_layer = layers.Conv2D(filters=n_class, kernel_size=1, strides=1, padding='same',
                                        activation=activation_last, name=output_name)(conv_level_3_to_4)
        
        y = layers.Flatten()(one_hot_layer)
        y = layers.Dense(2048, activation='relu')(y)
        y = layers.Dropout(0.5)(y)
        y = layers.Dense(2048, activation='relu')(y)
        y_0 = layers.Dropout(0.5)(y)
        y = layers.Lambda(lambda  x_: K.l2_normalize(y,axis=1))(y)
        y = layers.Dense(2, activation='softmax', name="classification")(y)
        '''
        if (reshape):
            #print(one_hot_layer.shape)
            one_hot_layer = keras.layers.Reshape(
                (self.input_shape[0]*self.input_shape[1], n_class), name=output_name)(one_hot_layer)
        return one_hot_layer#, y
    
    def CoreModel_old(self,x, class_name, n_filter=32,n_class=3, output_name="", reshape=True, kernel_size_softmax=1, activation_last="softmax", factor_kernel=1):
        
        #512x512 Level 3 --> 512x512x32
        conv_level_5 = self.Conv2DBNSLU(x=x, filters=n_filter, kernel_size=(1*factor_kernel,1*factor_kernel), strides=1,
                                padding='same', activation='selu', name='%s_conv_level_512x512x32'% (output_name)) #5,4
        
        #512x512x32 --> 256x256x64
        conv_level_4 = layers.Conv2D(filters=int(round(n_filter*2)), kernel_size=3*factor_kernel, strides=1,
                                padding='same', activation='selu', name= '%s_conv_level_256x256x64' %(output_name))(conv_level_5)
        conv_level_4 = layers.MaxPooling2D(pool_size=(2,2),strides=2)(conv_level_4) #5

        #256x256x64 --> 64x64x96
        conv_level_3 = layers.Conv2D(filters=int(round(n_filter*4)), kernel_size=3*factor_kernel, strides=1,
                            padding='same', activation='selu', name= '%s_conv_level_128x128x96' %(output_name))(conv_level_4)
        conv_level_3 = layers.MaxPooling2D(pool_size=(2,2),strides=2)(conv_level_3) #4
        
        #64x64x96 --> 16x16x128
        conv_level_2 = layers.Conv2D(filters=int(round(n_filter*8)), kernel_size=3*factor_kernel, strides=2,
                            padding='same', activation='selu', name= '%s_conv_level_16x16x128' %(output_name))(conv_level_3)
        conv_level_2 = layers.MaxPooling2D(pool_size=(2,2),strides=2)(conv_level_2) #4
        
        #16x16x128 --> 8x8x256
        conv_level_1 = layers.Conv2D(filters=int(round(n_filter*16)), kernel_size=3*factor_kernel, strides=1,
                            padding='same', activation='selu', name= '%s_conv_level_8x8x256' %(output_name))(conv_level_2)
        conv_level_1 = layers.MaxPooling2D(pool_size=(2,2),strides=2)(conv_level_1)

        #16x16 level 1 --> 8x8 (4,5)
        conv_level_1 = self.Conv2DBNSLU(x=conv_level_2, filters=int(round(n_filter*24)), kernel_size=(3*factor_kernel,3*factor_kernel), strides=1,
                                padding='same', activation='selu', name='%s_conv_level_1_0'% (output_name))
        conv_level_1 = layers.MaxPooling2D(pool_size=(2,2),strides=2)(conv_level_1)
        conv_level_1 = layers.SpatialDropout2D(0.25)(conv_level_1)

        #subgroups 
        conv_level_1_7 = layers.Conv2D(filters=int(round(n_filter*4)), kernel_size=7, strides=1, padding='same', activation='selu', name='%s_conv_level_1_7x7' %(output_name)) (conv_level_1)
        conv_level_1_7 = layers.Conv2D(filters=int(round(n_filter*4)), kernel_size=(1,7), strides=1, padding='same', activation='selu', name='%s_conv_level_1_1x7' %(output_name)) (conv_level_1_7)
        conv_level_1_7 = layers.Conv2D(filters=int(round(n_filter*4)), kernel_size=(7,1), strides=1, padding='same', activation='selu', name='%s_conv_level_1_7x1'% (output_name)) (conv_level_1_7)
        conv_level_1_7 = layers.SpatialDropout2D(0.25)(conv_level_1_7)
        
        conv_level_1_5 = layers.Conv2D(filters=int(round(n_filter*4)), kernel_size=5, strides=1, padding='same', activation='selu', name='%s_conv_level_1_0_5x5' % (output_name)) (conv_level_1)
        conv_level_1_5 = layers.Conv2D(filters=int(round(n_filter*4)), kernel_size=(5,1*factor_kernel), strides=1, padding='same', activation='selu', name='%s_conv_level_1_5x1' % (output_name)) (conv_level_1_5)
        conv_level_1_5 = layers.Conv2D(filters=int(round(n_filter*4)), kernel_size=(1*factor_kernel,5), strides=1, padding='same', activation='selu', name='%s_conv_level_1_1x5' % (output_name )) (conv_level_1_5)
        conv_level_1_5 = layers.SpatialDropout2D(0.25)(conv_level_1_5)

        conv_level_1_3 = layers.Conv2D(filters=int(round(n_filter*4)), kernel_size=3, strides=1, padding='same', activation='selu', name='%s_conv_level_1_3' %(output_name))(conv_level_1)
        conv_level_1_3 = layers.Conv2D(filters=int(round(n_filter*4)), kernel_size=(3,1*factor_kernel), strides=1, padding='same', activation='selu', name='%s_conv_level_1_1_3x1' % (output_name)) (conv_level_1_3)
        conv_level_1_3 = layers.Conv2D(filters=int(round(n_filter*4)), kernel_size=(1*factor_kernel, 3), strides=1,
                                       padding='same', activation='selu', name='%s_conv_level_1_2_1x3' % (output_name))(conv_level_1_3)
        conv_level_1_3 = layers.SpatialDropout2D(0.25)(conv_level_1_3)
        #Concatenate
        conv_level_1 = concatenate([conv_level_1,
                                    conv_level_1_5,
                                    conv_level_1_3,
                                    conv_level_1_7
                                    ])
        
        #8 -> 16
        upsample_level_0_to_1 = keras.layers.UpSampling2D()(conv_level_1)
        upsample_level_0_to_1 = self.Conv2DBNSLU(x=upsample_level_0_to_1, filters=int(round(n_filter*16)), kernel_size=(3*factor_kernel,3*factor_kernel), strides=1,
                                padding='same', activation='selu', name='%s_conv_level_16x16x_Up'% (output_name))
        conv_level_2 = layers.Conv2D(filters=int(round(n_filter*16)), kernel_size=3*factor_kernel, strides=1,
                            padding='same', activation='selu', name= '%s_conv_level_16x16xA_Up' %(output_name))(conv_level_2) #4
        conv_level_0_to_1 = concatenate([upsample_level_0_to_1, conv_level_2])
        conv_level_0_to_1 = layers.Conv2D(filters=int(round(n_filter*16)), kernel_size=3*factor_kernel, strides=1,
                            padding='same', activation='selu', name= '%s_conv_level_16x16xB_Up' %(output_name))(conv_level_0_to_1) #4
        
        #16 -> 64
        upsample_level_1_to_2 = keras.layers.UpSampling2D((4,4))(conv_level_0_to_1)
        upsample_level_1_to_2 = self.Conv2DBNSLU(x=upsample_level_1_to_2, filters=int(round(n_filter*8)), kernel_size=(3*factor_kernel,3*factor_kernel), strides=1,
                                padding='same', activation='selu', name='%s_conv_level_64x64x_Up'% (output_name)) #4
        conv_level_3 = layers.Conv2D(filters=int(round(n_filter*8)), kernel_size=3*factor_kernel, strides=1,
                            padding='same', activation='selu', name= '%s_conv_level_64x64xA_Up' %(output_name))(conv_level_3) #4
        conv_level_1_to_2 = concatenate([upsample_level_1_to_2, conv_level_3])
        conv_level_1_to_2 = layers.Conv2D(filters=int(round(n_filter*8)), kernel_size=3*factor_kernel, strides=1,
                            padding='same', activation='selu', name= '%s_conv_level_64x64xB_Up' %(output_name))(conv_level_1_to_2) #4
        
        ##64 -> 256
        upsample_level_2_to_3 = keras.layers.UpSampling2D((2,2))(conv_level_1_to_2)
        upsample_level_2_to_3 = self.Conv2DBNSLU(x=upsample_level_2_to_3, filters=int(round(n_filter*4)), kernel_size=(3*factor_kernel,3*factor_kernel), strides=1,
                                padding='same', activation='selu', name='%s_conv_level_256x256x_Up'% (output_name)) #3,3
        conv_level_4 = layers.Conv2D(filters=int(round(n_filter*3)), kernel_size=3*factor_kernel, strides=1,
                            padding='same', activation='selu', name= '%s_conv_level_256x256xA_Up' %(output_name))(conv_level_4) #4
        conv_level_2_to_3 = concatenate([upsample_level_2_to_3, conv_level_4])
        conv_level_2_to_3 = layers.Conv2D(filters=int(round(n_filter*2)), kernel_size=3*factor_kernel, strides=1,
                            padding='same', activation='selu', name= '%s_conv_level_256x256xB_Up' %(output_name))(conv_level_2_to_3)#4
        
        ##256 -> 512
        upsample_level_3_to_4 = keras.layers.UpSampling2D()(conv_level_2_to_3)
        upsample_level_3_to_4 = self.Conv2DBNSLU(x=upsample_level_3_to_4, filters=int(round(n_filter)), kernel_size=(3*factor_kernel,4*factor_kernel), strides=1,
                                padding='same', activation='selu', name='%s_conv_level_512x512x_Up'% (output_name)) #3,4
        conv_level_3_to_4 = concatenate([upsample_level_3_to_4, conv_level_5])
        conv_level_3_to_4 = layers.Conv2D(filters=int(round(n_filter)), kernel_size=3*factor_kernel, strides=1,
                            padding='same', activation='selu', name= '%s_conv_level_512x512xB_Up' %(output_name))(conv_level_3_to_4) #4
        

        #Softmax
        one_hot_layer_1 = layers.Conv2D(filters=n_class, kernel_size=1, strides=1,
                                        padding='same', activation=activation_last, name='%ss_ucnet11'%(output_name))(conv_level_3_to_4) #name='%s_ucnet01' % (output_name))(conv_level_3_to_4)
            
        one_hot_layer_2 = layers.Conv2D(filters=n_class, kernel_size=(5*factor_kernel,5*factor_kernel), strides=1,
                                    padding='same', activation=activation_last, name='%s_ucnet55'%(output_name))(conv_level_3_to_4)                
        
        one_hot_layer_3 = layers.Conv2D(filters=n_class, kernel_size=(3*factor_kernel,3*factor_kernel), strides=1,
                                    padding='same', activation=activation_last, name='%s_ucnet_33'%(output_name))(conv_level_3_to_4)
        #one_hot_layer_3 = layers.UpSampling2D(size=(4,4))(one_hot_layer_3)
        #Average layer
        one_hot_layer = keras.layers.average(
            [one_hot_layer_2, one_hot_layer_3, one_hot_layer_1], name=output_name)
        
        '''
        one_hot_layer = layers.Conv2D(filters=n_class, kernel_size=1, strides=1, padding='same',
                                        activation=activation_last, name=output_name)(conv_level_3_to_4)
        one_hot_layer = layers.Conv2D(filters=n_class, kernel_size=1, strides=1, padding='same',
                                        activation=activation_last, name=output_name)(conv_level_3_to_4)
        one_hot_layer = layers.Conv2D(filters=n_class, kernel_size=1, strides=1, padding='same',
                                        activation=activation_last, name=output_name)(conv_level_3_to_4)
        
        y = layers.Flatten()(one_hot_layer)
        y = layers.Dense(2048, activation='relu')(y)
        y = layers.Dropout(0.5)(y)
        y = layers.Dense(2048, activation='relu')(y)
        y_0 = layers.Dropout(0.5)(y)
        y = layers.Lambda(lambda  x_: K.l2_normalize(y,axis=1))(y)
        y = layers.Dense(2, activation='softmax', name="classification")(y)
        '''
        if (reshape):
            #print(one_hot_layer.shape)
            one_hot_layer = keras.layers.Reshape(
                (self.input_shape[0]*self.input_shape[1], n_class), name=output_name)(one_hot_layer)
        return one_hot_layer#, y
    

    def SegModel_Contour(self,x,output_name="", kernel_size_softmax=1, activation_last="sigmoid", n_class=1):
        #256
        conv_level_4 = layers.Conv2D(filters=16, kernel_size=3, strides=1, 
                                padding='same', activation='selu', name= '%s_conv_level_4_1_0' %(output_name))(x) #1
        conv_level_4 = layers.Conv2D(filters=16, kernel_size=3, strides=1, 
                                padding='same', activation='selu', name= '%s_conv_level_4_1_1' %(output_name))(x) #1
        
        #128
        conv_level_3 = layers.MaxPooling2D(pool_size=(2,2),strides=2)(conv_level_4)

        conv_level_3 = self.Conv2DBNSLU(x=conv_level_3, filters=32, kernel_size=3, strides=1,
                                padding='same', activation='selu', name='%s_conv_level_3_0_0'%(output_name), bias=True, bn=False) #4,2
        conv_level_3 = self.Conv2DBNSLU(x=conv_level_3, filters=32, kernel_size=3, strides=1,
                                padding='same', activation='selu', name='%s_conv_level_3_0_1'%(output_name), bias=True, bn=False) #4,2
       
        #64
        conv_level_2 = layers.MaxPooling2D(pool_size=(2,2),strides=2)(conv_level_3)

        conv_level_2 = self.Conv2DBNSLU(x=conv_level_2, filters=64, kernel_size=3, strides=1,
                                padding='same', activation='selu', name='%s_conv_level_2_0_0'%(output_name), bias=True, bn=False) #4,2
        conv_level_2 = self.Conv2DBNSLU(x=conv_level_2, filters=64, kernel_size=3, strides=1,
                                padding='same', activation='selu', name='%s_conv_level_2_0_1'%(output_name), bias=True, bn=False) #4,2
        conv_level_2 = self.Conv2DBNSLU(x=conv_level_2, filters=64, kernel_size=3, strides=1,
                                padding='same', activation='selu', name='%s_conv_level_2_0_2'%(output_name), bias=True, bn=False) #4,2
       
        #32
        conv_level_1 = layers.MaxPooling2D(pool_size=(2,2),strides=2)(conv_level_2)

        conv_level_1 = self.Conv2DBNSLU(x=conv_level_1, filters=128, kernel_size=3, strides=1,
                                padding='same', activation='selu', name='%s_conv_level_1_0_0'%(output_name), bias=True, bn=False) #4,2
        conv_level_1 = self.Conv2DBNSLU(x=conv_level_1, filters=128, kernel_size=3, strides=1,
                                padding='same', activation='selu', name='%s_conv_level_1_0_1'%(output_name), bias=True, bn=False) #4,2
        conv_level_1 = self.Conv2DBNSLU(x=conv_level_1, filters=128, kernel_size=3, strides=1,
                                padding='same', activation='selu', name='%s_conv_level_1_0_2'%(output_name), bias=True, bn=False) #4,2

        #16
        conv_level_0 = layers.MaxPooling2D(pool_size=(2,2),strides=2)(conv_level_1)

        conv_level_0 = self.Conv2DBNSLU(x=conv_level_0, filters=256, kernel_size=3, strides=1,
                                padding='same', activation='selu', name='%s_conv_level_0_0_0'%(output_name), bias=True, bn=False) #4,2
        conv_level_0 = self.Conv2DBNSLU(x=conv_level_0, filters=256, kernel_size=3, strides=1,
                                padding='same', activation='selu', name='%s_conv_level_0_0_1'%(output_name), bias=True, bn=False) #4,2
        conv_level_0 = self.Conv2DBNSLU(x=conv_level_0, filters=256, kernel_size=3, strides=1,
                                padding='same', activation='selu', name='%s_conv_level_0_0_2'%(output_name), bias=True, bn=False) #4,2

        #8
        conv_level_0_0 = layers.MaxPooling2D(pool_size=(2,2),strides=2)(conv_level_0)

        conv_level_0_0 = self.Conv2DBNSLU(x=conv_level_0_0, filters=512, kernel_size=3, strides=1,
                                padding='same', activation='selu', name='%s_conv_level_00_0_0'%(output_name), bias=True, bn=False) #4,2
        conv_level_0_0 = self.Conv2DBNSLU(x=conv_level_0_0, filters=512, kernel_size=3, strides=1,
                                padding='same', activation='selu', name='%s_conv_level_00_0_1'%(output_name), bias=True, bn=False) #4,2
        conv_level_0_0 = self.Conv2DBNSLU(x=conv_level_0_0, filters=512, kernel_size=3, strides=1,
                                padding='same', activation='selu', name='%s_conv_level_00_0_2'%(output_name), bias=True, bn=False) #4,2
        #Decoder
        #8
        conv_level_0_1 = self.Conv2DBNSLU(x=conv_level_0_0, filters=512, kernel_size=1, strides=1,
                                        padding='same', activation='selu', name='%s_conv_level_0_0_0_0' % (output_name), bias=True, bn=False) #2
        #16
        conv_level_0_2 = keras.layers.Conv2DTranspose(filters=256, kernel_size=(3,3), strides=2, padding="same")(conv_level_0_1)
        #32
        conv_level_0_3 = keras.layers.Conv2DTranspose(filters=128, kernel_size=(3,3), strides=2, padding="same")(conv_level_0_2)
        #64
        conv_level_0_4 = keras.layers.Conv2DTranspose(filters=64, kernel_size=(3,3), strides=2, padding="same")(conv_level_0_3) #4,4
        #128
        conv_level_0_5 = keras.layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=2, padding="same")(conv_level_0_4) #2,2
        #256
        conv_level_0_6 = keras.layers.Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=2, padding="same")(conv_level_0_5) #2,2

        #conv_level_1 = sub([conv_level_0_0,conv_level_4])
        #Sigmoid
        one_hot_layer = layers.Conv2D(filters=n_class, kernel_size=kernel_size_softmax, strides=1,
                                        padding='same', activation=activation_last, name=output_name)(conv_level_0_6)
        return one_hot_layer


    def SegModel(self,x,output_name="", kernel_size_softmax=1, activation_last="sigmoid", n_class=1):
        #256
        conv_level_4 = layers.Conv2D(filters=16, kernel_size=3, strides=1, 
                                padding='same', activation='selu', name= '%s_conv_level_4_0' %(output_name))(x) #1
        #128
        conv_level_3 = self.Conv2DBNSLU(x=conv_level_4, filters=32, kernel_size=(3,3), strides=2,
                                padding='same', activation='selu', name='%s_conv_level_3_0'%(output_name)) #4,2
        #64
        #conv_level_3 = layers.MaxPooling2D(pool_size=(2,2),strides=2)(conv_level_3)
        conv_level_3 = self.Conv2DBNSLU(x=conv_level_3, filters=64, kernel_size=(3, 3), strides=2,
                                padding='same', activation='selu', name='%s_conv_level_3_1' %(output_name)) #2,4
        conv_level_3 = layers.SpatialDropout2D(0.2)(conv_level_3)
        #32
        conv_level_3 = self.Conv2DBNSLU(x=conv_level_3, filters=128, kernel_size=(3, 3), strides=2,
                                padding='same', activation='selu', name='%s_conv_level_3_3'%(output_name)) #2,4
        
        #16
        conv_level_2 = self.Conv2DBNSLU(x=conv_level_3, filters=256, kernel_size=3, strides=2,
                                        padding='same', activation='selu', name='%s_conv_level_0_0_1_0' % (output_name)) #2
        conv_level_2 = layers.SpatialDropout2D(0.3)(conv_level_2) #
        #8
        '''
        conv_level_2 = self.Conv2DBNSLU(x=conv_level_2, filters=512, kernel_size=3, strides=2,
                                        padding='same', activation='selu', name='%s_conv_level_0_0_0_0' % (output_name)) #2
        conv_level_1 = keras.layers.Conv2DTranspose(filters=256, kernel_size=(3,3), strides=2, padding="same")(conv_level_2)
        '''
        conv_level_1 = keras.layers.Conv2DTranspose(filters=128, kernel_size=(3,3), strides=2, padding="same")(conv_level_2)
        #16s
        conv_level_1 = keras.layers.Conv2DTranspose(filters=64, kernel_size=(3,3), strides=2, padding="same")(conv_level_1) #4,4
        #32
        conv_level_0 = keras.layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=2, padding="same")(conv_level_1) #2,2
        #64
        conv_level_0_0 = keras.layers.Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=2, padding="same")(conv_level_0) #2,2

        conv_level_1 = conv_level_0_0 #concatenate([conv_level_0_0,conv_level_4])
        #Sigmoid
        one_hot_layer = layers.Conv2D(filters=n_class, kernel_size=kernel_size_softmax, strides=1,
                                        padding='same', activation=activation_last, name=output_name)(conv_level_1)
        return one_hot_layer
    

