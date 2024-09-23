
    
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
d=10


cnt=0
for model_name in models:
    cnt=cnt+1
    print (cnt,' ',len (models),'  ',model_name)
    for i in A:
        if model_name.lower() in i.lower():
            input_string = i
            percentage = float(re.search(r'(\d+\.\d+)%', input_string).group(1))
            backbone_name = input_string.split()[0]
            backbone_names.append(backbone_name)
            accuracies.append(percentage)

            model = eval(model_name)
            f1, f2 = model_MCC(model)
            times.append(float(f1))
            parameters.append(model.count_params())
            mcc = 0.01 * (percentage**a) * (c+ np.log(d + float(f1)))**(b)
            mcc = round(mcc, 2)
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
max_acc_index = accuracies.index(max(accuracies))
max_acc_backbone = backbone_names[max_acc_index]

# Find maximum MCC and corresponding backbone name
max_mcc_index = mccs.index(max(mccs))
max_mcc_backbone = backbone_names[max_mcc_index]

# Plotting
plt.close('all')
plt.figure(figsize=(10, 6))
plt.suptitle(  'Accuracy and MCC by Model', fontname='Times New Roman')
plt.subplot(211)
plt.plot(accuracies,'*', label='Accuracy (%)')
# plt.xticks(ticks=range(len(backbone_names)), labels=backbone_names, rotation=45, fontname='Times New Roman')  # Set font to Times New Roman
plt.xticks(ticks=range(len(backbone_names)), labels=range(1, len(backbone_names) + 1), rotation=45, fontname='Times New Roman')  # Set font to Times New Roman
# plt.xlabel('Backbone Names', fontname='Times New Roman')  # Set font to Times New Roman
plt.ylabel('Accuracy Value', fontname='Times New Roman')  # Set font to Times New Roman
# plt.title('Accuracy by Model', fontname='Times New Roman')  # Set font to Times New Roman
plt.legend(prop={'family': 'Times New Roman'})  # Set font of legend text to Times New Roman
plt.grid(True)

# Annotate maximum ACC and MCC values
plt.annotate(f'Max ACC: {max(accuracies)} ({max_acc_backbone})', 
             xy=(max_acc_index, max(accuracies)), 
             xytext=(-20, 20), 
             textcoords='offset points', 
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
             fontname='Times New Roman')  # Set font to Times New Roman

plt.subplot(212)
plt.plot(mccs,'*', label='MCC (%)')  # Change label to indicate MCC is in percentage
plt.xlabel('Backbone Names', fontname='Times New Roman')  # Set font to Times New Roman
plt.ylabel('MCC Value', fontname='Times New Roman')  # Set font to Times New Roman
# plt.title(' MCC by Model', fontname='Times New Roman')  # Set font to Times New Roman
plt.xticks(ticks=range(len(backbone_names)), labels=backbone_names, rotation=45, fontname='Times New Roman')  # Set font to Times New Roman
plt.legend(prop={'family': 'Times New Roman'})  # Set font of legend text to Times New Roman
plt.grid(True)


plt.annotate(f'Max MCC: {max(mccs)} ({max_mcc_backbone})', 
             xy=(max_mcc_index, max(mccs)), 
             xytext=(-20, -30), 
             textcoords='offset points', 
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
             fontname='Times New Roman')  # Set font to Times New Roman

plt.show()
