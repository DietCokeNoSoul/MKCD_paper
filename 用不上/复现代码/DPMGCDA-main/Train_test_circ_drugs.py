'''
任务:采用CNN来对circRNA-drugs数据进行预测,并通过无责交叉验证,然后输出AUC和AUPR
初始数据集:circRNA_drugs_association、circRNA_seq_sim、drugs_str_sim
需要计算获得的数据:circrna_gip_sim、drug_gip_sim、circrna_snf_sim、drug_snf_sim
'''


# 导入第三方包
import pandas as pd
import torch
import os
import sys
import dgl
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.data import Data
from tqdm.auto import tqdm
# custom modules
from maskgae.model import DPMGCDA, EdgeDecoder, GNNEncoder, FeatureExtracter
from maskgae.mask import MaskPath
import data_loader


# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到 sys.path
sys.path.append(project_root)

# 导入自定义包
from data_loader import MyDataset
from util import plt
import clac_metric


torch.cuda.empty_cache()
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

'''
    数据加载与预处理
'''
# 读取数据 circRNA-drug*271*218 覆盖过得特征矩阵5*489*489(cd, cc_snf_sim, dd_snf_sim) 训练集索引表5*[[2, 3304],[2, 3304]] 测试集索引表5*[[2, 826],[2, 50814]]
cd, features, trainSet_index, testSet_index = torch.load('DPMGCDA_1_1.pth')


'''
    超参数设置
'''
epoch = 600
learn_rate = 0.001
batch_size = 2**16

'''
    创建模型
'''
def create_model(ifGraph=False):
    mask = MaskPath(p=0.7, num_nodes=271 + 218, start="edge", walk_length=2)
    edge_decoder = EdgeDecoder(128, 64, num_layers=2, dropout=0.2)
    
    if ifGraph:
        feature_extracter = FeatureExtracter([489, 489], [256, 256])
        encoder = GNNEncoder(256, 128, 128, num_layers=1, dropout=0.2)
    else:
        feature_extracter = None
        encoder = GNNEncoder(489, 128, 128, num_layers=1, dropout=0.2)

    model = DPMGCDA(encoder, edge_decoder, mask, feature_extracter).to(device)

    return model

'''
    训练模型
'''

for times in range(1):  # 5折交叉验证
    cth = 0.0040
    dth = 0.0048
    # 结点数
    c_node_num = 271
    d_node_num = 218
    mean_fprs, mean_tprs, mean_recalls, mean_precs= [], [], [], []
    for i in range(5):  # 五折交叉
        # 从featues中取出cc相似度矩阵和dd相似度矩阵，featues[i]是一个特征矩阵，前271行和前271列是cc相似度矩阵，后218行和后218列是dd相似度矩阵
        c_snf_sim = features[i][:271, :271]  # 271*271
        d_snf_sim = features[i][271:, 271:]  # 218*218
        # 从featues中取出cda特征矩阵,featues[i]是一个特征矩阵，前271行和后218列是cda特征矩阵
        cda = features[i][:271, 271:]  # circRNA-drugs的关联矩阵，271*218，行为circRNA，列为disease，值为是否存在关联边
        # 根据cth和dth二值化构建同构图
        c_snf_sim_bin = (c_snf_sim > cth).float()
        d_snf_sim_bin = (d_snf_sim > dth).float()
        src, dst = c_snf_sim_bin.nonzero(as_tuple=True)
        homo_graph_c = dgl.graph((src, dst))
        src, dst = d_snf_sim_bin.nonzero(as_tuple=True)
        homo_graph_d = dgl.graph((src, dst))
        homo_graph = [homo_graph_c.to(device), homo_graph_d.to(device)]  # 271*271 218*218     
           
        # 构建节点特征矩阵
        c_feature = torch.cat((c_snf_sim, cda), dim=1)  # 271*489 circRNA图初始特征矩阵,行为circRNA,列为特征
        d_feature = torch.cat((d_snf_sim, cda.t()), dim=1)  # 218*489 disease图初始特征矩阵,行为disease,列为特征
        
        d_graph_feature = torch.randn((d_node_num, 489))
        c_graph_feature = torch.randn((c_node_num, 489))
        
        node_snf_features = [c_feature.to(device), d_feature.to(device)]
        node_graph_features = [c_graph_feature.to(device), d_graph_feature.to(device)]
        
        # 使用trainSet_index和testSet_index划分数据集
        train_edges_pos = torch.cat([trainSet_index[i][0], testSet_index[i][0]], dim =1)
        train_edges = train_edges_pos
        
        
        train_data_neg = torch.cat([trainSet_index[i][1], testSet_index[i][1]], dim =1)
        train_edges_test = [train_edges_pos, train_data_neg] #[2, 4130] [2, 4130]
        
        
        test_edges_pos = testSet_index[i][0]
        test_edges_neg = testSet_index[i][1]
        test_edges = [test_edges_pos, test_edges_neg] # [2, 826] [2, 826]
        
        train_data = Data(edge_index=train_edges).to(device)
        test_data = Data(edge_index=train_edges).to(device)

        model_f = create_model(False)  # 没有feature_extracter
        model_g = create_model(True)  # 有feature_extracter
        optimizer_f = torch.optim.Adam(model_f.parameters(), lr=learn_rate, weight_decay=5e-5)
        optimizer_g = torch.optim.Adam(model_g.parameters(), lr=learn_rate, weight_decay=5e-5)
         
        print("Combined Feature Level is training")
        for epoch in range(1, 1 + epoch):
            loss = model_f.train_step(train_data, optimizer_f, homo_graph, node_snf_features, batch_size=batch_size)
            if epoch % 30 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
        print("Homogeneous Graph Level is training")
        for epoch in range(1, 1 + epoch):
            loss = model_g.train_step(train_data, optimizer_g, homo_graph, node_graph_features, batch_size=batch_size)
            if epoch % 30 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
        y, f_pred = model_f.test_step(test_data, test_edges, homo_graph, node_snf_features, batch_size=2**16)
        y, g_pred = model_g.test_step(test_data, test_edges, homo_graph, node_graph_features, batch_size=2**16)
        # y, f_pred = model_f.test_step(train_data, train_edges, homo_graph, node_snf_features,batch_size=2**16)
        # y, g_pred = model_g.test_step(train_data, train_edges, homo_graph, node_graph_features,batch_size=2**16)
        pred = (f_pred + g_pred) / 2 # 两视角结果取平均
        y = y.cpu().numpy()
        pred = pred.cpu().numpy()
        metric_tmp = clac_metric.get_metrics(y, pred)
        print("aupr, auc, f1_score, accuracy, recall, specificity, precision")
        print(metric_tmp)

    
        

print('-----------------------------------------------------------------------finish-----------------------------------------------------------------------')

