import tensorflow as tf
from typing import Tuple, List

class CardUNet:
    def __init__(self, input_shape : Tuple[int], backbone : str, n_classes : int) -> None:
        self.input_shape = input_shape
        self.backbone = backbone
        self.n_classes = n_classes
    def build_backbone(self, input_tensor : tf.keras.layers.Layer):
        if self.backbone == 'resnet50':
            resnet = tf.keras.applications.ResNet50(
                include_top=False,
                weights='imagenet',
                input_tensor=input_tensor
            )
            
            try:
                resnet = tf.keras.applications.ResNet50(
                    include_top=False,
                    weights='none',
                    input_tensor=input_tensor
                )
                
                resnet.load_weights('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
            except:
                print ('base model cant load weights')
            
            
            for layer in resnet.layers:
                layer.trainable = False
                
            for l in resnet.layers:
                l.trainable = True
                l.trainable = False
                
            out = resnet.get_layer('conv2_block2_out').output
            
            
            total_params = resnet.count_params()
            total_params1 = "{:,}".format(total_params)
            print("Number of  BB ResNet50V2 all Par :", total_params1,' <== Parameters')
            print("Back_bone shape:", resnet.output_shape)
            
            trainable_params = sum(tf.keras.backend.count_params(w) for w in resnet.trainable_variables)
            untrainable_params = total_params - trainable_params
            total_params1 = "{:,}".format(total_params)
            trainable_params1 = "{:,}".format(trainable_params)
            untrainable_params1 = "{:,}".format(untrainable_params)
         
            print(10*'-')
            print("Number of  All         BB ResNet50V2 using Par:", total_params1,' <== Parameters')
            print("Number of    Trainable BB ResNet50V2 using Par:", trainable_params1,' <== Parameters')
            print("Number of  UnTrainable BB ResNet50V2 using Par:", untrainable_params1,' <== Parameters')
            
        return out
    def down_block(self, x, filters, use_maxpool = True, *, name):
        x = tf.keras.layers.Conv2D(filters, 3, padding= 'same', name=f'conv_1_downblock_{name}', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(name=f'batchnorm_1_downblock_{name}')(x)
        x = tf.keras.layers.LeakyReLU(name=f'leakyrelu_1_downblock_{name}')(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding= 'same', name=f'conv_2_downblock_{name}', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(name=f'batchnorm_2_downblock_{name}')(x)
        x = tf.keras.layers.LeakyReLU(name=f'leakyrelu_2_downblock_{name}')(x)
        if use_maxpool == True:
            return  tf.keras.layers.MaxPooling2D(strides= (2,2), name=f'maxpooling_downblock_{name}')(x), x
        else:
            return x
    def up_block(self, x, y, filters, *, name):
        x = tf.keras.layers.UpSampling2D(name=f'upsample_upblock_{name}')(x)
        x = tf.keras.layers.Concatenate(axis = 3, name=f'concate_upblock_{name}')([x,y])
        x = tf.keras.layers.Conv2D(filters, 3, padding= 'same', name=f'conv_1_upblock_{name}', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(name=f'batchnorm_1_upblock_{name}')(x)
        x = tf.keras.layers.LeakyReLU(name=f'leakyrelu_1_upblock_{name}')(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding= 'same', name=f'conv_2_upblock_{name}', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(name=f'batchnorm_2_upblock_{name}')(x)
        x = tf.keras.layers.LeakyReLU(name=f'leakyrelu_2_upblock_{name}')(x)
        return x
    def build_mcme(self, input_tensor : tf.keras.layers.Layer, skip_connections : List[tf.Tensor]):
        vgg19 = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet',
            input_tensor=input_tensor
        )
        for layer in vgg19.layers:
            layer.trainable = False
        output_vgg19 = vgg19.output
        x = tf.keras.layers.Reshape((256, 256, 2))(output_vgg19)
        x = tf.keras.layers.Conv2DTranspose(6, 1, activation='softmax')(x)
        G1 = x[:, :, :, 0]
        G2 = x[:, :, :, 1]
        G3 = x[:, :, :, 2]
        G4 = x[:, :, :, 3]
        G5 = x[:, :, :, 4]
        G6 = x[:, :, :, 5]

        G1 = tf.keras.layers.Conv2DTranspose(3, 1, activation='sigmoid')(G1[:, :, :, None])
        G2 = tf.keras.layers.Conv2DTranspose(3, 1, activation='sigmoid')(G2[:, :, :, None])
        G3 = tf.keras.layers.Conv2DTranspose(3, 1, activation='sigmoid')(G3[:, :, :, None])
        G4 = tf.keras.layers.Conv2DTranspose(3, 1, activation='sigmoid')(G4[:, :, :, None])
        G5 = tf.keras.layers.Conv2DTranspose(3, 1, activation='sigmoid')(G5[:, :, :, None])
        G6 = tf.keras.layers.Conv2DTranspose(3, 1, activation='sigmoid')(G6[:, :, :, None])

        a1 = tf.keras.layers.multiply([G1, skip_connections[0]], name="a1")
        a2 = tf.keras.layers.multiply([G2, skip_connections[1]], name="a2")
        a3 = tf.keras.layers.multiply([G3, skip_connections[2]], name="a3")
        a4 = tf.keras.layers.multiply([G4, skip_connections[3]], name="a4")
        a5 = tf.keras.layers.multiply([G5, skip_connections[4]], name="a5")
        a6 = tf.keras.layers.multiply([G6, skip_connections[5]], name="a6")

        return a1, a2, a3, a4, a5, a6

    def build_model(self):
        # input
        input_tensor = tf.keras.layers.Input(self.input_shape, name="input_tensor")
        resizing= tf.keras.layers.Resizing(512, 512, name="resizing_input")(input_tensor)
        print ('+'*50)
        print ('input shape        =',input_tensor.shape)
        # backbone
        out_backbone_resnet = self.build_backbone(resizing) 
        E1= tf.keras.layers.Resizing(256, 256, name="resizing_input_E1")(input_tensor) # E1
        # encode
        out_con_128, out_128 = self.down_block(out_backbone_resnet, 128, name='128') # E2
        out_conv_256, out_256 = self.down_block(out_con_128, 256, name='256') # E3
        out_conv_512, out_512 = self.down_block(out_conv_256, 512, name='512') # E4
        out_conv_1024, out_1024 = self.down_block(out_conv_512, 1024, name='1024') # E5
        # bottleneck
        bottleneck = self.down_block(out_conv_1024, 2048, use_maxpool= False, name='2048') # E6
        # decode
        x = self.up_block(bottleneck, out_1024, 1024, name='1024')
        x = self.up_block(x, out_512, 512, name='512')
        x = self.up_block(x, out_256, 256, name='256')
        x = self.up_block(x, out_128, 128, name='128')
        x = tf.keras.layers.UpSampling2D(name=f'upsample_final')(x)
        x = tf.keras.layers.Conv2DTranspose(3, 1, activation='linear')(x)
        # output
        
        print(f"E1: {E1.shape}")
        print(f"E2: {out_128.shape}")
        print(f"E3: {out_256.shape}")
        print(f"E4: {out_512.shape}")
        print(f"E5: {out_1024.shape}")
        print(f"E6: {bottleneck.shape}")

        E1 = tf.keras.layers.Conv2DTranspose(3, 1, activation='sigmoid')(tf.keras.layers.Resizing(256, 256)(E1))
        E2 = tf.keras.layers.Conv2DTranspose(3, 1, activation='sigmoid')(tf.keras.layers.Resizing(256, 256)(out_128))
        E3 = tf.keras.layers.Conv2DTranspose(3, 1, activation='sigmoid')(tf.keras.layers.Resizing(256, 256)(out_256))
        E4 = tf.keras.layers.Conv2DTranspose(3, 1, activation='sigmoid')(tf.keras.layers.Resizing(256, 256)(out_512))
        E5 = tf.keras.layers.Conv2DTranspose(3, 1, activation='sigmoid')(tf.keras.layers.Resizing(256, 256)(out_1024))
        E6 = tf.keras.layers.Conv2DTranspose(3, 1, activation='sigmoid')(tf.keras.layers.Resizing(256, 256)(bottleneck))

        # print(E1.shape, E2.shape, E3.shape, E4.shape, E5.shape, E6.shape)

        a1, a2, a3, a4, a5, a6 = self.build_mcme(resizing, skip_connections=[E1, E2, E3, E4, E5, E6])

        # print(a1.shape, a2.shape, a3.shape, a4.shape, a5.shape, a6.shape)

        AMM_1 = a1 + a2 + a3 + a4 + a5 + a6
        AMM_2 = tf.keras.layers.Concatenate(axis=-1, name="concatenate_AMM")([AMM_1, x])
        
        # print(AMM_2.shape)
        
        output = tf.keras.layers.Conv2D(self.n_classes, 1, activation= 'softmax', name="output")(AMM_2)
        output1 = tf.keras.layers.Conv2D(self.n_classes, 1, activation= 'softmax', name="output")(x)
        print(output.shape, output1.shape)
        # print(output.dtype, output1.dtype)
        # print(output, output1)
        # model
        model = tf.keras.Model(inputs=input_tensor, outputs=output)
        
        
        total_params = model.count_params()
        trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_variables)
        
        untrainable_params = total_params - trainable_params
        total_params1 = "{:,}".format(total_params)
        trainable_params1 = "{:,}".format(trainable_params)
        untrainable_params1 = "{:,}".format(untrainable_params)
     
        print(10*'-')
        print("Number of  All         MECardNet  using Par:", total_params1,' <== Parameters')
        print("Number of    Trainable MECardNet  using Par:", trainable_params1,' <== Parameters')
        print("Number of  UnTrainable MECardNet  using Par:", untrainable_params1,' <== Parameters')
        print(10*'-')
        
        

        return model
    
    


show_example = True
# show_example = False

if show_example : 
    cardunet = CardUNet( input_shape=(256, 256, 3), backbone='resnet50',    n_classes=3  )
    MECardNet = cardunet.build_model()
    # model.summary()