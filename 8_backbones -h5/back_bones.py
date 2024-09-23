

import tensorflow as tf

def param_model (model):
    # print(10*'*')
    total_params = model.count_params()
    total_params1 = "{:,}".format(total_params)
    # print(10*'-')
    # print("Back_bone input_shape :", model.input_shape)
    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_variables)
    untrainable_params = total_params - trainable_params
    total_params1 = "{:,}".format(total_params)
    trainable_params1 = "{:,}".format(trainable_params)
    untrainable_params1 = "{:,}".format(untrainable_params)
    s1='';s2='';s3=''
    for i in range(10-len(str(total_params1 ))):  s1=s1+' '
    for i in range(10-len(str(trainable_params1 ))):  s2=s2+' '
    for i in range(10-len(str(untrainable_params1 ))):  s3=s3+' '
    print("Number of  All            Par:", total_params1,s1,' <== Parameters')
    # print("Number of  Trainable      Par:", trainable_params1,s2,' <== Parameters')
    # print("Number of  UnTrainable    Par:", untrainable_params1,s3,' <== Parameters')
    print("Back_bone output_shape:", model.output_shape)
    # print(10*'-')
    temp=0
    
    best_layer=[];best_layer2=[]
    ibest_layer=[];ibest_layer2=[]
    
    print('len(model.layers) =',len(model.layers))
    # Print the names and sizes of all layers
    for i, layer in enumerate(model.layers):
        output_shape = layer.output_shape
        # print(i, layer.name, output_shape)
        size=64
        try:
            if output_shape[1]==size and temp==0:
                temp=1;#print('start')
                best_layer.append(layer.name)
                ibest_layer.append(i)
            if output_shape[1]==size and temp==1:
                 temp=1;#print('continue')
                 best_layer.append(layer.name)
                 ibest_layer.append(i)
            if output_shape[1]<=size and temp==1:
                 temp=2;#print('stop')
            if output_shape[1]==size and temp==2:
                 temp=2;#print('section2')
                 best_layer2.append(layer.name)
                 ibest_layer2.append(i)
        except:pass
    # model1 = model.get_layer(best_layer[-1]).output
    # model2 = model.get_layer(best_layer2[-1]).output
    
    # layer_name = ibest_layer[-1]  # Replace 'your_layer_name' with the actual name of the layer

    # # Get the layer by its name
    # layer = model.get_layer(layer_name)
    
    # # Get the shape of the layer's output
    # output_shape = layer.output_shape
    # print("Output shape of layer {}: {}".format(layer_name, output_shape))


    print(best_layer2[-1],' index 1 = ' ,ibest_layer[-1], ' from ' ,len(model.layers),'layer')
    print(best_layer2[-1],'index 2 = ' ,ibest_layer2[-1], ' from ' ,len(model.layers),'layer')
    print(1*'\n')
    
    try:
    model1_model = tf.keras.Model(inputs=convnext_tiny.inputs, outputs=model1)
    total_params = sum(p.shape.num_elements() for p in model1_model.trainable_weights)
    
    model2_model = tf.keras.Model(inputs=convnext_tiny.inputs, outputs=model2)
    total_params2 = sum(p.shape.num_elements() for p in model2_model.trainable_weights)
# except:
    # total_params = model1.count_params()
    # total_params2 = model2.count_params()
    total_params1 = "{:,}".format(total_params)
    
    s1='';s2='';s3=''
    for i in range(10-len(str(total_params1 ))):  s1=s1+' '
    for i in range(10-len(str(total_params2 ))):  s2=s2+' '
    print("Number of  model1            Par:", total_params1,s1,' <== Parameters index layer=',ibest_layer[-1])    
        
    
    total_params2 = "{:,}".format(total_params2)
    print("Number of  model2            Par:", total_params2,s2,' <== Parameters index layer=',ibest_layer2[-1])    
      
input_shape = (256, 256, 3)
input_tensor = tf.keras.layers.Input(input_shape, name="input_tensor")



convnext_base = tf.keras.applications.convnext.ConvNeXtBase(
    model_name='convnext_base',
    include_top=True,    include_preprocessing=True,
    weights= None ,    input_tensor=input_tensor,    input_shape=None,
    pooling=None,    classes=1000,    classifier_activation='softmax')


convnext_large = tf.keras.applications.convnext.ConvNeXtLarge(
    model_name='convnext_large',
    include_top=True,
    include_preprocessing=True,
    weights=None,    input_tensor=input_tensor,    input_shape=None,
    pooling=None,    classes=1000,    classifier_activation='softmax'
)


convnext_small=tf.keras.applications.convnext.ConvNeXtSmall(
    model_name='convnext_small',
    include_top=True,    include_preprocessing=True,
    weights=None,    input_tensor=input_tensor,
    input_shape=None,
    pooling=None,    classes=1000,    classifier_activation='softmax'
)


convnext_tiny=tf.keras.applications.convnext.ConvNeXtTiny(
    model_name='convnext_tiny',    include_top=True,    include_preprocessing=True,
    weights=None,    input_tensor=input_tensor,    input_shape=None,
    pooling=None,    classes=1000,    classifier_activation='softmax'
)

convnext_xlarge = tf.keras.applications.convnext.ConvNeXtXLarge(
    model_name='convnext_xlarge',
    include_top=True,    include_preprocessing=True,
    weights=None,    input_tensor=input_tensor,    input_shape=None,    pooling=None,    classes=1000,
    classifier_activation='softmax'
)


# eff_model=tf.keras.applications.efficientnet()
# eff_model2=tf.keras.applications.efficientnet_v2 ()


print(10*'\n')
print("Back_bone input_shape :(None, 256, 256, 3)" )
print('convnext_tiny') ; param_model (convnext_tiny);
print('convnext_small') ; param_model (convnext_small);
print('convnext_base') ; param_model (convnext_base);
print('convnext_large') ; param_model (convnext_large);
print('convnext_xlarge') ; param_model (convnext_xlarge);



if False :
    model = convnext_small
    
    temp=0
    # Print the names and sizes of all layers
    for i, layer in enumerate(model.layers):
        output_shape = layer.output_shape
        print(i, layer.name, output_shape)
        best_layer=[]
        try:
            if output_shape[1]==64 and temp==0:
                temp=1;print('start')
                best_layer.append(layer.name)
            if output_shape[1]==64 and temp==1:
                  temp=1;print('continue')
                  best_layer.append(layer.name)
            if output_shape[1]<=64 and temp==1:
                  temp=2;print('stop')
            if output_shape[1]==64 and temp==2:
                  temp=2;print('section2')
        except:pass

    model = model
    temp=0
    
    best_layer=[];best_layer2=[]
    ibest_layer=[];ibest_layer2=[]
    
    print('len(model.layers) =',len(model.layers))
    # Print the names and sizes of all layers
    for i, layer in enumerate(model.layers):
        output_shape = layer.output_shape
        # print(i, layer.name, output_shape)
        
        try:
            if output_shape[1]==64 and temp==0:
                temp=1;#print('start')
                best_layer.append(layer.name)
                ibest_layer.append(i)
            if output_shape[1]==64 and temp==1:
                  temp=1;#print('continue')
                  best_layer.append(layer.name)
                  ibest_layer.append(i)
            if output_shape[1]<=64 and temp==1:
                  temp=2;#print('stop')
            if output_shape[1]==64 and temp==2:
                  temp=2;#print('section2')
                  best_layer2.append(layer.name)
                  ibest_layer2.append(i)
        except:pass
    
    model1 = model.get_layer(best_layer[-1]).output
    model2 = model.get_layer(best_layer2[-1]).output
    
    layer_name1 = best_layer[-1]
    layer_name2 = best_layer2[-1]
    
    model1 = convnext_tiny.get_layer(layer_name1).output
    model2 = convnext_tiny.get_layer(layer_name2).output
    model2_model = tf.keras.Model(inputs=convnext_tiny.inputs, outputs=model2)
    # Calculate the total number of parameters in the model
    total_params = sum(p.shape.num_elements() for p in model2_model.trainable_weights)
    print("Total parameters:", total_params)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    