plt.annotate(f'min Time ( Second) : {min(times)} ({min_times_backbone})', 
             xy=(min_times_index, min(times)), 
             xytext=( -60, -60), 
             textcoords='offset points', 
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
             fontname='Times New Roman')  # Set font to Times New Roman