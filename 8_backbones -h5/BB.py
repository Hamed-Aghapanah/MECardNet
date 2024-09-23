
model_loader =False
# model_loader =True
model_MCC_calc =False
# model_MCC_calc =True
# model_ploter =False
model_ploter =True



import tensorflow as tf
import keras
# from keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1 for INFO, 2 for WARNING, 3 for ERROR)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (DEBUG, INFO, WARNING, ERROR)
import time

# =============================================================================
# functions
# =============================================================================
def parameters_model (model):
    total_params = model.count_params()
    # total_params1 = "{:,}".format(total_params)
    total_params1 = "{:,}".format(total_params)
    return total_params1


def weight_finder (f1,f2,f3='NONE',f4='NONE'):
    file_path=None 
    import os
    current_directory = os.getcwd()
    files = os.listdir(current_directory)
    
    filtered_files=[]
    for i in files:
        i2 = i.lower()
        # print (i2)
        if (f1.lower() in i2)  and (f2.lower() in i2) and not(f3.lower() in i2) and not(f4.lower() in i2) :
            filtered_files.append (i)
    # print ('****',filtered_files)
    for file_name in filtered_files:
        s='';
        while len (file_name) +len(s) <30:
            s=s+' '
        print(file_name,s,'++')
        file_path =   file_name  
    if len (filtered_files) <1:
        print(f1,f2 , 'Not find   ---------')
        file_path=None 
        file_path='imagenet' 
    # print ('file_path =' , file_path)
    return file_path

def timer_predicted(model):
    import time  # Importing the time module here
    img_path = 'saed.jpg'  # Change this to the path of your image
    input_shape = model.input_shape

    # print("Input shape:", input_shape)
    height = input_shape[1]
    width = input_shape[2]
    channels = input_shape[3] if len(input_shape) == 4 else None  # Check if the shape includes the channel dimension

    img = keras.utils.load_img(img_path, target_size=(height, width))
    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.resnet50.preprocess_input(x)  # Preprocess input specific to ResNet50
    
    start_time = time.time()  # Using the time module to get current time
    
    with tf.device('/CPU:0'):  # Ensure predictions happen on CPU to avoid GPU logging
        preds = model.predict(x, verbose=0)
        
    elapsed_time = time.time() - start_time  # Calculate elapsed time
    
    elapsed_time = round(elapsed_time, 2) 
    
    # print("Elapsed time:", elapsed_time, "seconds")
    return elapsed_time



def model_MCC (model):
    time1 = timer_predicted(model)
    total_params1 = parameters_model(model)
    return   time1 ,total_params1

  
    




if model_loader :
    
    
    
    
    
          # ConvNeXtBase family 
    input_shape = (256, 256, 3)
    input_tensor = tf.keras.layers.Input(input_shape, name="input_tensor")
    
    # 'MobileNet ', 
    # 'MobileNetV2',
    
    w =weight_finder ('MobileNet','MobileNet','V2')
   

    MobileNet = tf.keras.applications.MobileNet(
    include_top=True, 
    # include_preprocessing=True,
    weights=w,  # Assuming you want ImageNet weights. Replace 'imagenet' with your variable 'w' if different.
    input_tensor=input_tensor, input_shape=None,  # Specify input shape if you want, e.g., (224, 224, 3) for MobileNet.
    pooling=None,     classes=1000,      classifier_activation='softmax'
    )
    
    w =weight_finder ('MobileNet','V2' )
    MobileNetV2 = tf.keras.applications.MobileNetV2(
    include_top=True, 
    # include_preprocessing=True,
    weights=w,  # Assuming you want ImageNet weights. Replace 'imagenet' with your variable 'w' if different.
    input_tensor=input_tensor, input_shape=None,  # Specify input shape if you want, e.g., (224, 224, 3) for MobileNet.
    pooling=None,     classes=1000,      classifier_activation='softmax'
    )
    
    w =weight_finder ('convnext','tiny')
    ConvNeXtTiny=tf.keras.applications.convnext.ConvNeXtTiny(
        # model_name='ConvNeXtTiny',
        include_top=True,    include_preprocessing=True,
        weights=w,    input_tensor=input_tensor,        input_shape=None,
        pooling=None,    classes=1000,    classifier_activation='softmax'
    )
    
    w =weight_finder ('convnext','small')
    
    ConvNeXtSmall=tf.keras.applications.convnext.ConvNeXtSmall(
        # model_name='ConvNeXtSmall',
        include_top=True,    include_preprocessing=True,
        weights=w,    input_tensor=input_tensor,    
        input_shape=None,
        pooling=None,    classes=1000,    classifier_activation='softmax'
    )
    w =weight_finder ('convnext','_base')
    ConvNeXtBase = tf.keras.applications.convnext.ConvNeXtBase(
        # model_name='ConvNeXtBase',
        include_top=True,    include_preprocessing=True,
        weights= w ,    input_tensor=input_tensor,    input_shape=None,
        pooling=None,    classes=1000,    classifier_activation='softmax'
        )
    
    
    w =weight_finder ('convnext','large','xlarge')
    ConvNeXtLarge = tf.keras.applications.convnext.ConvNeXtLarge(
        # model_name='ConvNeXtLarge',
        include_top=True,
        include_preprocessing=True,
        weights=w,    input_tensor=input_tensor,        input_shape=None,
        pooling=None,    classes=1000,    classifier_activation='softmax'
    )
    
    
    w =weight_finder ('convnext','xlarge')
    ConvNeXtXLarge = tf.keras.applications.convnext.ConvNeXtXLarge(
        # model_name='ConvNeXtXLarge',
        include_top=True,    include_preprocessing=True,
        weights=w,    input_tensor=input_tensor,        input_shape=None,    pooling=None,    classes=1000,
        classifier_activation='softmax'
    )
    
    
    
    # efficientnet family
    w =weight_finder ('EfficientNet','B0','v2')
    EfficientNetB0 = tf.keras.applications.EfficientNetB0(weights=w,    input_tensor=input_tensor,)
    w =weight_finder ('EfficientNet','B1','v2')
    EfficientNetB1 = tf.keras.applications.EfficientNetB1(weights=w,    input_tensor=input_tensor,)
    w =weight_finder ('EfficientNet','B2','v2')
    EfficientNetB2 = tf.keras.applications.EfficientNetB2(weights=w,    input_tensor=input_tensor,)
    w =weight_finder ('EfficientNet','B3','v2')
    EfficientNetB3 = tf.keras.applications.EfficientNetB3(weights=w,    input_tensor=input_tensor,)
    w =weight_finder ('EfficientNet','B4','v2')
    EfficientNetB4 = tf.keras.applications.EfficientNetB4(weights=w,    input_tensor=input_tensor,)
    w =weight_finder ('EfficientNet','B5','v2')
    EfficientNetB5 = tf.keras.applications.EfficientNetB5(weights=w,    input_tensor=input_tensor,)
    w =weight_finder ('EfficientNet','B6','v2')
    EfficientNetB6 = tf.keras.applications.EfficientNetB6(weights=w,    input_tensor=input_tensor,)
    w =weight_finder ('EfficientNet','B7','v2')
    EfficientNetB7 = tf.keras.applications.EfficientNetB7(weights=w,    input_tensor=input_tensor,)
    
    w =weight_finder ('efficientnetv2','B0')
    EfficientNetV2B0 = tf.keras.applications.EfficientNetV2B0(weights=w,    input_tensor=input_tensor,)
    w =weight_finder ('efficientnetv2','B1')
    EfficientNetV2B1 = tf.keras.applications.EfficientNetV2B1(weights=w,    input_tensor=input_tensor,)
    w =weight_finder ('efficientnetv2','B2')
    EfficientNetV2B2 = tf.keras.applications.EfficientNetV2B2(weights=w,    input_tensor=input_tensor,)
    w =weight_finder ('efficientnetv2','B3')
    EfficientNetV2B3 = tf.keras.applications.EfficientNetV2B3(weights=w,    input_tensor=input_tensor,)
    w =weight_finder ('efficientnetv2','-s')
    EfficientNetV2S = tf.keras.applications.EfficientNetV2S(weights=w,    input_tensor=input_tensor,)
    w =weight_finder ('efficientnetv2','-m')
    EfficientNetV2M = tf.keras.applications.EfficientNetV2M(weights=w,    input_tensor=input_tensor,)
    w =weight_finder ('efficientnetv2','-l')
    EfficientNetV2L = tf.keras.applications.EfficientNetV2L(weights=w,    input_tensor=input_tensor,)
     
    
    
    
    # VGG family
    w =weight_finder ('vgg','16')
    VGG16 = tf.keras.applications.VGG16( weights=None )
    w =weight_finder ('vgg','19','notop')
    VGG19 = tf.keras.applications.VGG19(weights=w,)
    
    
    # Inception family
    
    w =weight_finder ('inception','v3')
    InceptionV3 = tf.keras.applications.InceptionV3(weights=w,    input_tensor=input_tensor,)
    w =weight_finder ('inception','resnet')
    InceptionResNetV2 = tf.keras.applications.InceptionResNetV2(weights=w,    input_tensor=input_tensor,)
    
    
    # xception
    w =weight_finder ('Xception','Xception')
    Xception = tf.keras.applications.Xception(weights=w,    input_tensor=input_tensor,)
    
    
    # NASNet family
    w =weight_finder ('NASNet','large')
    NASNetLarge = tf.keras.applications.NASNetLarge(weights=w)
    
    w =weight_finder ('NASNet','Mobile')
    NASNetMobile = tf.keras.applications.NASNetMobile(weights=w)
    
    
    # ResNet family
    w =weight_finder ('ResNet','50','v2','notop')
    ResNet50 = tf.keras.applications.ResNet50(weights=w,    input_tensor=input_tensor,)
    
    w =weight_finder ('ResNet50V2','50','notop')
    ResNet50V2 = tf.keras.applications.ResNet50V2(weights=w,    input_tensor=input_tensor,)
    
    w =weight_finder ('ResNet101','101','v2')
    ResNet101 = tf.keras.applications.ResNet101(weights=w,    input_tensor=input_tensor,)
    
    w =weight_finder ('ResNet','101v2')
    ResNet101V2 = tf.keras.applications.ResNet101V2(weights=w,    input_tensor=input_tensor,)
    
    w =weight_finder ('ResNet','152','V2')
    ResNet152 = tf.keras.applications.ResNet152(weights=w,    input_tensor=input_tensor,)
    
    w =weight_finder ('ResNet152','V2')
    ResNet152V2 = tf.keras.applications.ResNet152V2(weights=w,    input_tensor=input_tensor,)
    
    
    # densenet family
    w =weight_finder ('DenseNet','121')
    DenseNet121 = tf.keras.applications.DenseNet121(weights=w,    input_tensor=input_tensor, include_top=True)
    w =weight_finder ('DenseNet','169')
    DenseNet169 = tf.keras.applications.DenseNet169(weights=w,    input_tensor=input_tensor, include_top=True)
    w =weight_finder ('DenseNet','201')
    DenseNet201 = tf.keras.applications.DenseNet201(weights=w,    input_tensor=input_tensor, include_top=True)
    
    

if model_MCC_calc :
    print(10*'\n')
    print("Back_bone input_shape :(None, 256, 256, 3)" )
    print(1*'\n')
    
    
    
    models = [
        'MobileNet ', 
        'MobileNetV2',
               'ConvNeXtTiny',
               'ConvNeXtSmall',
                'ConvNeXtBase',
                'ConvNeXtLarge',
                 'ConvNeXtXLarge',
              
                'EfficientNetB0',
                'EfficientNetB1','EfficientNetB2',
                'EfficientNetB3','EfficientNetB4','EfficientNetB5','EfficientNetB6','EfficientNetB7',
                'EfficientNetV2B0',
                'EfficientNetV2B1',
                'EfficientNetV2B2',
                'EfficientNetV2B3','EfficientNetV2S','EfficientNetV2M','EfficientNetV2L' ,
              
              
               'VGG16','VGG19',
                'InceptionV3','InceptionResNetV2',
              
               'Xception',
              
                'NASNetLarge',
                'NASNetMobile',
               'ResNet50', 
                'ResNet101','ResNet152',
                'ResNet50V2', 'ResNet101V2','ResNet152V2',
              
               'DenseNet121', 'DenseNet169', 
              'DenseNet201'
              ]
    
    
    A = [
     'Xception 	88 	79.0% 	94.5% 	22.9M 	81 	109.4 	8.1' , 
     'VGG16 	528 	71.3% 	90.1% 	138.4M 	16 	69.5 	4.2' , 
    'VGG19 	549 	71.3% 	90.0% 	143.7M 	19 	84.8 	4.4' , 
    'ResNet50 	98 	74.9% 	92.1% 	25.6M 	107 	58.2 	4.6' , 
    'ResNet50V2 	98 	76.0% 	93.0% 	25.6M 	103 	45.6 	4.4' , 
    'ResNet101 	171 	76.4% 	92.8% 	44.7M 	209 	89.6 	5.2' , 
    'ResNet101V2 	171 	77.2% 	93.8% 	44.7M 	205 	72.7 	5.4' , 
    'ResNet152 	232 	76.6% 	93.1% 	60.4M 	311 	127.4 	6.5' , 
    'ResNet152V2 	232 	78.0% 	94.2% 	60.4M 	307 	107.5 	6.6' , 
    
    'InceptionV3 	92 	77.9% 	93.7% 	23.9M 	189 	42.2 	6.9' , 
    'InceptionResNetV2 	215 	80.3% 	95.3% 	55.9M 	449 	130.2 	10.0' , 
    'MobileNet 	16 	70.4% 	89.5% 	4.3M 	55 	22.6 	3.4' , 
    'MobileNetV2 	14 	71.3% 	90.1% 	3.5M 	105 	25.9 	3.8' , 
    'DenseNet121 	33 	75.0% 	92.3% 	8.1M 	242 	77.1 	5.4' , 
    'DenseNet169 	57 	76.2% 	93.2% 	14.3M 	338 	96.4 	6.3' , 
    'DenseNet201 	80 	77.3% 	93.6% 	20.2M 	402 	127.2 	6.7' , 
    'NASNetMobile 	23 	74.4% 	91.9% 	5.3M 	389 	27.0 	6.7' , 
    'NASNetLarge 	343 	82.5% 	96.0% 	88.9M 	533 	344.5 	20.0' , 
    
    'EfficientNetB0 	29 	77.1% 	93.3% 	5.3M 	132 	46.0 	4.9' , 
    'EfficientNetB1 	31 	79.1% 	94.4% 	7.9M 	186 	60.2 	5.6' , 
    'EfficientNetB2 	36 	80.1% 	94.9% 	9.2M 	186 	80.8 	6.5' , 
    'EfficientNetB3 	48 	81.6% 	95.7% 	12.3M 	210 	140.0 	8.8' , 
    'EfficientNetB4 	75 	82.9% 	96.4% 	19.5M 	258 	308.3 	15.1' , 
    'EfficientNetB5 	118 	83.6% 	96.7% 	30.6M 	312 	579.2 	25.3' , 
    'EfficientNetB6 	166 	84.0% 	96.8% 	43.3M 	360 	958.1 	40.4' , 
    'EfficientNetB7 	256 	84.3% 	97.0% 	66.7M 	438 	1578.9 	61.6' , 
    'EfficientNetV2B0 	29 	78.7% 	94.3% 	7.2M 	- 	- 	-' , 
    'EfficientNetV2B1 	34 	79.8% 	95.0% 	8.2M 	- 	- 	-' , 
    'EfficientNetV2B2 	42 	80.5% 	95.1% 	10.2M 	- 	- 	-' , 
    'EfficientNetV2B3 	59 	82.0% 	95.8% 	14.5M 	- 	- 	-' , 
    'EfficientNetV2S 	88 	83.9% 	96.7% 	21.6M 	- 	- 	-' , 
    'EfficientNetV2M 	220 	85.3% 	97.4% 	54.4M 	- 	- 	-' , 
    'EfficientNetV2L 	479 	85.7% 	97.5% 	119.0M 	- 	- 	-' , 
    
    'ConvNeXtTiny 	109.42 	81.3% 	- 	28.6M 	- 	- 	-' , 
    'ConvNeXtSmall 	192.29 	82.3% 	- 	50.2M 	- 	- 	-' , 
    'ConvNeXtBase 	338.58 	85.3% 	- 	88.5M 	- 	- 	-' , 
    'ConvNeXtLarge 	755.07 	86.3% 	- 	197.7M 	- 	- 	-' , 
    'ConvNeXtXLarge 	1310 	86.7% 	- 	350.1M 	- 	- 	- ',
    ]
    
     
# Sort the list based on the first word in each element
sorted_A = sorted(A, key=lambda x: x.split()[0])


# Sort the list based on the first word in each element
sorted_models = sorted(models, key=lambda x: x.split()[0])

    
if model_ploter:    
    import re
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Initialize lists to store data
    backbone_names = []
    times = []
    parameters = []
    accuracies = []
    mccs = []
    
    # Loop through models and A list
    a=2
    b=-1
    c=10
    d=20
    
    
    cnt=0
    for model_name in sorted_models:
        cnt=cnt+1
        print (cnt,' ',len (models),'  ',model_name)
        for i in sorted_A:
            if model_name.lower() in i.lower():
                input_string = i
                percentage = float(re.search(r'(\d+\.\d+)%', input_string).group(1))
                backbone_name = input_string.split()[0]
                backbone_names.append(backbone_name)
                accuracies.append(percentage)
                
                f1 =10
                f2=0
    
                # if not 'ConvNeXt' in i :
                model = eval(model_name)
                f1, f2 = model_MCC(model)
                times.append(float(f1))
                parameters.append(model.count_params())
                mcc = 0.01 * (percentage**a) * (c+ np.log(d + float(f1)))**(b)
                mcc = np.log (1+percentage/100/np.log  (f1+100))
                mcc = np.log (1+percentage/100)
                mcc=(mcc-0.3)/(0.8-0.3)
                mcc=percentage
                timeee = f1 *15*25 
                mcc = round(mcc, 2)
                if timeee >60*5 :
                    mcc=0
                
                
                
                mccs.append(mcc)
                break  # If found the model, exit the loop
    
    # Print the headers
    print("   Back_bone name     Time         Param.     ACC       Model Competence Criterion (MCC)" )
    print(80*"-" )
    # Loop through data and print
    # mccs=(max (accuracies) - min (accuracies))*( mccs-min (mccs)) /(max(mccs) - min (mccs)) +min (accuracies)
    for cnt, (backbone_name, time, param, acc, mcc) in enumerate(zip(backbone_names, times, parameters, accuracies, mccs), 1):
        print(f"{cnt:<2} {backbone_name:<20} {time:<10.2f} { round (param/1000000,1):<10} {acc:<10} {mcc:<10}")
    
    
    
    # Find maximum ACC and corresponding backbone name
    max_times_index = times.index(max(times))
    max_times_backbone = backbone_names[max_times_index]
    
    min_times_index = times.index(min(times))
    min_times_backbone = backbone_names[min_times_index]
    
    
    max_acc_index = accuracies.index(max(accuracies))
    min_acc_index = accuracies.index(min(accuracies))
    max_acc_backbone = backbone_names[max_acc_index]
    min_acc_backbone = backbone_names[min_acc_index]
    
    # Find maximum MCC and corresponding backbone name
    max_mcc_index = mccs.index(max(mccs))
    max_mcc_backbone = backbone_names[max_mcc_index]
    
    # Plotting
    plt.close('all')
    import matplotlib.pyplot as plt
    import tkinter as tk
    
    # Get screen resolution
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    
    # Create a fullscreen figure
    plt.figure(figsize=(screen_width/100, screen_height/100))
    
    sss=15
    # plt.figure(figsize=(10, 6))
    # plt.suptitle(  'Accuracy and MCC by Model', fontname='Times New Roman')
    plt.subplot(311)
    plt.plot(accuracies,'k*', label='Dice (%)')
    # plt.xticks(ticks=range(len(backbone_names)), labels=backbone_names, rotation=45, fontname='Times New Roman')  # Set font to Times New Roman
    plt.yticks(fontname='Times New Roman')
    plt.xticks(ticks=range(len(backbone_names)), labels=range(1, len(backbone_names) + 1), rotation=45, fontname='Times New Roman')  # Set font to Times New Roman
    # plt.xlabel('Backbone Names', fontname='Times New Roman')  # Set font to Times New Roman
    plt.ylabel('Dice Value', fontname='Times New Roman')  # Set font to Times New Roman
    # plt.title('Accuracy by Model', fontname='Times New Roman')  # Set font to Times New Roman
    plt.legend(prop={'family': 'Times New Roman'})  # Set font of legend text to Times New Roman
    plt.grid(True)
    
    # Annotate maximum ACC and MCC values
    plt.annotate(f'Max dice: {max(accuracies)} ({max_acc_backbone})', 
                 xy=(max_acc_index, max(accuracies)), 
                 xytext=( -20,  -20), 
                 textcoords='offset points', 
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2",color='green'),
                 fontname='Times New Roman',color='green',size =sss)  # Set font to Times New Roman
    plt.annotate(f'Min dice: {max(accuracies)} ({min_acc_backbone})', 
                 xy=(min_acc_index, min(accuracies)), 
                 xytext=( +20,  +20), 
                 textcoords='offset points', 
                 arrowprops=dict(arrowstyle="->",color='red', connectionstyle="arc3,rad=.2"),
                 fontname='Times New Roman',color='red',size =sss)  # Set font to Times New Roman
    
    plt.subplot(312)
    plt.plot(times,'c*', label='Time (Second)')  # Change label to indicate MCC is in percentage
    # plt.xlabel('Backbone Names', fontname='Times New Roman')  # Set font to Times New Roman
    plt.ylabel('Time Value', fontname='Times New Roman')  # Set font to Times New Roman
    # plt.title(' MCC by Model', fontname='Times New Roman')  # Set font to Times New Roman
    # plt.xticks(ticks=range(len(backbone_names)), labels=backbone_names, rotation=45, fontname='Times New Roman')  # Set font to Times New Roman
    plt.xticks(ticks=range(len(backbone_names)), labels=range(1, len(backbone_names) + 1), rotation=45, fontname='Times New Roman')  # Set font to Times New Roman
    plt.yticks(fontname='Times New Roman')

    plt.legend(prop={'family': 'Times New Roman'})  # Set font of legend text to Times New Roman
    plt.grid(True)
    plt.annotate(f'Max Time (Second) : {max(times)} ({max_times_backbone})', 
                 xy=(max_times_index, max(times)), 
                 xytext=( -20,  -20), 
                 textcoords='offset points', 
                 arrowprops=dict(arrowstyle="->",color='red', connectionstyle="arc3,rad=.2"),
                 fontname='Times New Roman',color='red',size =sss)  # Set font to Times New Roman
    
    plt.annotate(f'Min Time (Second) : {min(times)} ({min_times_backbone})', 
                 xy=(min_times_index, min(times)), 
                 xytext=( +20, +20), 
                 textcoords='offset points', 
                 arrowprops=dict(arrowstyle="->",color='green', connectionstyle="arc3,rad=.2"),
                 fontname='Times New Roman',color='green',size =sss)  # Set font to Times New Roman
    
    plt.subplot(313)
    plt.plot(mccs, 'm*', label='MCC')  # Change label to indicate MCC is in percentage
    plt.xlabel('Backbone Names', fontname='Times New Roman')  # Set font to Times New Roman
    plt.ylabel('MCC Value', fontname='Times New Roman')  # Set font to Times New Roman
    # plt.title('MCC by Model', fontname='Times New Roman')  # Set font to Times New Roman
    plt.xticks(ticks=range(len(backbone_names)), labels=backbone_names, rotation=45, fontname='Times New Roman')  # Set font to Times New Roman
    plt.yticks(fontname='Times New Roman')

    # Recolor xticks with value 0 to red
    xticks_labels = plt.gca().get_xticklabels()
    for index in range (len (xticks_labels)):
        if int(mccs[index]) == 0:
            xticks_labels[index].set_color('red')
    
    plt.legend(prop={'family': 'Times New Roman'})  # Set font of legend text to Times New Roman
    plt.grid(True)
    filtered_mcc = [num for num in mccs if num != 0]

    # Find the minimum value in the filtered list
    min_value = min(filtered_mcc)-1
    max_value = max(filtered_mcc)+1

    plt.ylim([min_value,max_value])
    
    plt.annotate(f'Max MCC: {max(mccs)} ({max_mcc_backbone})', 
                 xy=(max_mcc_index, max(mccs)), 
                 xytext=(-40, -40), 
                 textcoords='offset points', 
                 arrowprops=dict(arrowstyle="->",color='green', connectionstyle="arc3,rad=.2"),
                 fontname='Times New Roman',color='green',size =sss)  # Set font to Times New Roman
    plt.savefig('accuracy_mcc_plot2.png', bbox_inches='tight', dpi=300)  # Save as PNG with tight bounding box
    plt.show()



    
# os.system("shutdown /s /t 1")
