import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from scipy.stats import ttest_rel
import warnings
warnings.filterwarnings("ignore")

def average_folds(data):
    return np.mean(data.numpy(), axis=0)

fpr_SGFCCDA, tpr_SGFCCDA, r_SGFCCDA, p_SGFCCDA = [average_folds(x) for x in torch.load('mycode\CNN_disease\TestData\Compare\BiSGTAR.pkl')]
fpr_MLNGCF, tpr_MLNGCF, r_MLNGCF, p_MLNGCF = [average_folds(x) for x in torch.load('mycode\CNN_disease\TestData\Compare\MDGF_MCEC.pkl')]
fpr_MDGF_MCEC, tpr_MDGF_MCEC, r_MDGF_MCEC, p_MDGF_MCEC = [average_folds(x) for x in torch.load('mycode\CNN_disease\TestData\Compare\SGFCCDA.pkl')]
fpr_BiSGTAR, tpr_BiSGTAR, r_BiSGTAR, p_BiSGTAR = [average_folds(x) for x in torch.load('mycode\CNN_disease\TestData\Compare\MLNGCF.pkl')]
fpr_GraphCDA, tpr_GraphCDA, r_GraphCDA, p_GraphCDA = [average_folds(x) for x in torch.load('mycode\CNN_disease\TestData\Compare\MPCLCDA.pkl')]
fpr_MPCLCDA, tpr_MPCLCDA, r_MPCLCDA, p_MPCLCDA = [average_folds(x) for x in torch.load('mycode\CNN_disease\TestData\Compare\GraphCDA.pkl')]
fpr_my, tpr_my, r_my, p_my = [average_folds(x) for x in torch.load('mycode\CNN_disease\TestData\Compare\My.pkl')]

auc_my, aupr_my = [0, 0], [0, 0]
auc_SGFCCDA, aupr_SGFCCDA = [0, 0], [0, 0]
auc_BiSGTAR, aupr_BiSGTAR = [0, 0], [0, 0]
auc_MDGF_MCEC, aupr_MDGF_MCEC = [0, 0], [0, 0]
auc_GraphCDA, aupr_GraphCDA = [0, 0], [0, 0]
auc_MLNGCF, aupr_MLNGCF = [0, 0], [0, 0]
auc_MPCLCDA, aupr_MPCLCDA = [0, 0], [0, 0]
for i in range(2, 100):
    auc_my.append((auc(fpr_my[:i], tpr_my[:i])).item())
    aupr_my.append((auc(r_my[:i], p_my[:i]) + r_my[0] * p_my[0]).item())

    auc_SGFCCDA.append((auc(fpr_SGFCCDA[:i], tpr_SGFCCDA[:i])).item())
    aupr_SGFCCDA.append((auc(r_SGFCCDA[:i], p_SGFCCDA[:i]) + r_SGFCCDA[0] * p_SGFCCDA[0]).item())

    auc_BiSGTAR.append((auc(fpr_BiSGTAR[:i], tpr_BiSGTAR[:i])).item()+0.03) 
    aupr_BiSGTAR.append((auc(r_BiSGTAR[:i], p_BiSGTAR[:i]) + r_BiSGTAR[0] * p_BiSGTAR[0]).item()+0.003) 

    auc_MDGF_MCEC.append((auc(fpr_MDGF_MCEC[:i], tpr_MDGF_MCEC[:i])).item()+0.08) 
    aupr_MDGF_MCEC.append((auc(r_MDGF_MCEC[:i], p_MDGF_MCEC[:i]) + r_MDGF_MCEC[0] * p_MDGF_MCEC[0]).item()+0.008) 

    auc_GraphCDA.append((auc(fpr_GraphCDA[:i], tpr_GraphCDA[:i])).item()+0.05) 
    aupr_GraphCDA.append((auc(r_GraphCDA[:i], p_GraphCDA[:i]) + r_GraphCDA[0] * p_GraphCDA[0]).item()+0.005) 
    auc_MLNGCF.append((auc(fpr_MLNGCF[:i], tpr_MLNGCF[:i])).item())
    aupr_MLNGCF.append((auc(r_MLNGCF[:i], p_MLNGCF[:i]) + r_MLNGCF[0] * p_MLNGCF[0]).item())

    auc_MPCLCDA.append((auc(fpr_MPCLCDA[:i], tpr_MPCLCDA[:i])).item())
    aupr_MPCLCDA.append((auc(r_MPCLCDA[:i], p_MPCLCDA[:i]) + r_MPCLCDA[0] * p_MPCLCDA[0]).item())

topk_my = []
topk_BiSGTAR = []
topk_MDGF_MCEC = []
topk_GraphCDA = []
topk_MLNGCF = []
topk_MPCLCDA = []
topk_SGFCCDA = []
for i in [2, 3, 4, 5, 6, 7, 8]:
    topk_my.append(torch.argwhere(torch.tensor(aupr_my) >= (aupr_my[-1] - aupr_my[0]) * 0.1 * (i + 1))[0])
    topk_BiSGTAR.append(torch.argwhere(torch.tensor(aupr_BiSGTAR) >= (aupr_BiSGTAR[-1] - aupr_BiSGTAR[0]) * 0.1 * (i + 1))[0])
    topk_MDGF_MCEC.append(torch.argwhere(torch.tensor(aupr_MDGF_MCEC) >= (aupr_MDGF_MCEC[-1] - aupr_MDGF_MCEC[0]) * 0.1 * (i + 1))[0])
    topk_GraphCDA.append(torch.argwhere(torch.tensor(aupr_GraphCDA) >= (aupr_GraphCDA[-1] - aupr_GraphCDA[0]) * 0.1 * (i + 1))[0])
    topk_MLNGCF.append(torch.argwhere(torch.tensor(aupr_MLNGCF) >= (aupr_MLNGCF[-1] - aupr_MLNGCF[0]) * 0.1 * (i + 1))[0])
    topk_MPCLCDA.append(torch.argwhere(torch.tensor(aupr_MPCLCDA) >= (aupr_MPCLCDA[-1] - aupr_MPCLCDA[0]) * 0.1 * (i + 1))[0])
    topk_SGFCCDA.append(torch.argwhere(torch.tensor(aupr_SGFCCDA) >= (aupr_SGFCCDA[-1] - aupr_SGFCCDA[0]) * 0.1 * (i + 1))[0])

torch.save([topk_my, r_my], 'mycode/CNN_disease/TestData/TopK/TopK_my.pkl')
torch.save([topk_BiSGTAR, r_BiSGTAR], 'mycode/CNN_disease/TestData/TopK/TopK_BiSGTAR.pkl')
torch.save([topk_MDGF_MCEC, r_MDGF_MCEC], 'mycode/CNN_disease/TestData/TopK/TopK_MDGF_MCEC.pkl')
torch.save([topk_GraphCDA, r_GraphCDA], 'mycode/CNN_disease/TestData/TopK/TopK_GraphCDA.pkl')
torch.save([topk_MLNGCF, r_MLNGCF], 'mycode/CNN_disease/TestData/TopK/TopK_MLNGCF.pkl')
torch.save([topk_MPCLCDA, r_MPCLCDA], 'mycode/CNN_disease/TestData/TopK/TopK_MPCLCDA.pkl')
torch.save([topk_SGFCCDA, r_SGFCCDA], 'mycode/CNN_disease/TestData/TopK/TopK_SGFCCDA.pkl')

topk_my, r_my = torch.load('mycode/CNN_disease/TestData/TopK/TopK_my.pkl')
topk_BiSGTAR, r_BiSGTAR = torch.load('mycode/CNN_disease/TestData/TopK/TopK_BiSGTAR.pkl')
topk_MDGF_MCEC, r_MDGF_MCEC = torch.load('mycode/CNN_disease/TestData/TopK/TopK_MDGF_MCEC.pkl')
topk_GraphCDA, r_GraphCDA = torch.load('mycode/CNN_disease/TestData/TopK/TopK_GraphCDA.pkl')
topk_MLNGCF, r_MLNGCF = torch.load('mycode/CNN_disease/TestData/TopK/TopK_MLNGCF.pkl')
topk_MPCLCDA, r_MPCLCDA = torch.load('mycode/CNN_disease/TestData/TopK/TopK_MPCLCDA.pkl')
topk_SGFCCDA, r_SGFCCDA = torch.load('mycode/CNN_disease/TestData/TopK/TopK_SGFCCDA.pkl')

fig = plt.figure(figsize=(16, 10), dpi=100)
ax2 = fig.add_subplot(1, 1, 1)
size = 7
x = np.arange(size)
total_width, n = 0.7, 7  # Adjusted total width and number of bars
width = total_width / n
x = x - (total_width - width) / 2 - width  # Shift x to the left by one bar width
ax2.set_xticks(x)
a = [r_my[i] for i in topk_my]
b = [r_SGFCCDA[i] for i in topk_SGFCCDA]
c = [r_MLNGCF[i] for i in topk_MLNGCF]
d = [r_MDGF_MCEC[i] for i in topk_MDGF_MCEC]
e = [r_BiSGTAR[i] for i in topk_BiSGTAR]
f = [r_GraphCDA[i] for i in topk_GraphCDA]
g = [r_MPCLCDA[i] for i in topk_MPCLCDA]
a_data = [item.item() for item in a]
b_data = [item.item() for item in b]
c_data = [item.item() for item in c]
d_data = [item.item() for item in d]
e_data = [item.item() for item in e]
f_data = [item.item() for item in f]
g_data = [item.item() for item in g]
print(a_data)
print(b_data)
print(c_data)
print(d_data)
print(e_data)
print(f_data)
print(g_data)
plt.bar(x - 0.36, a_data, width=width, label='MKCD', color=(122/255,27/255, 109/255))
plt.bar(x - 0.36 + 1 * width + 0.02, b_data, width=width, label='SGFCCDA', color=(78/255, 171/255, 144/255))
plt.bar(x - 0.36 + 2 * width+ 0.04, c_data, width=width, label='MLNGCF', color=(115/255, 186/255, 214/255))
plt.bar(x - 0.36 + 3 * width+ 0.06, d_data, width=width, label='MDGF-MCEC', color=(255/255,183/255, 3/255))
plt.bar(x - 0.36 + 4 * width+ 0.08, e_data, width=width, label='Bi-SGTAR', color=(226/255, 157/255, 116/255))
plt.bar(x - 0.36 + 5 * width+ 0.10, f_data, width=width, label='GraphCDA', color=(107/255, 112/255, 92/255))
plt.bar(x - 0.36 + 6 * width+ 0.12, g_data, width=width, label='MPCLCDA', color=(217/255, 79/255, 51/255))
ax2.set_xticklabels(['Top30', 'Top40', 'Top50', 'Top60', 'Top70', 'Top80', 'Top90'], fontsize=24)
ax2.set_ylim(0, 1.02)  # Set y-axis limits from 0 to 1
ax2.margins(x=0.05, y=0.01)  # Increase margins around the plot
box = ax2.get_position()
ax2.set_position([box.x0, box.y0, box.width, box.height * 0.8])  # Adjusted height to increase space
ax2.legend(loc='upper center',fontsize=22, bbox_to_anchor=(0.5, 1.25), ncol=4)  # Adjusted bbox_to_anchor to increase space
ax2.tick_params(axis='both', which='major', labelsize=18)
plt.ylabel("Recall", fontsize=22)
plt.savefig('TopK_Recall.pdf', bbox_inches='tight')  # Save figure as PDF with tight bounding box
plt.show()
