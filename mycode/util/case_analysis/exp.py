import torch
import xlsxwriter as xw
# 所有药物top20候选微生物excel表，

def xw_toExcel(index_top, fileName):  # xlsxwriter库储存数据到excel
    workbook = xw.Workbook(fileName)  # 创建工作簿
    worksheet1 = workbook.add_worksheet("sheet1")  # 创建子表
    worksheet1.activate()  # 激活表
    title = ['drug_name', 'microbe_name']  # 设置表头
    worksheet1.write_row('A1', title)  # 从A1单元格开始写入表头
    i = 2  # 从第二行开始写入数据

    microbe_names = []
    with open('./data/microbe_names.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            microbe_names.append(line.rstrip())

    drug_names = []
    with open('./data/drug_names.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            drug_names.append(line.rstrip().strip('\ufeff'))

    for m in range(index_top.shape[0]):
        for n in range(index_top.shape[1]):
            insertData = [drug_names[m], microbe_names[index_top[m][n]]]

            row = 'A' + str(i)
            worksheet1.write_row(row, insertData)
            i += 1
    workbook.close()  # 关闭表


# 展示某一个药物的候选20个微生物
def show(index):
    microbe_names = []
    with open('./data/microbe_names.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            microbe_names.append(line.rstrip())
    for i in index:
        print(microbe_names[i])


result, y = torch.load('./result/result_pre_all')
# print(result950807.shape) torch.Size([237529, 2])
pred = torch.softmax(result, dim=1)[:, 1]
_, _, test_index, _, _, _, _ = torch.load('./embed_index_adj_nomusked.pth')

score = torch.zeros(size=(1373, 173))
i = 0
# 将预测的分值放入对应的位置
for x, y in test_index:
    score[x][y] = pred[i]
    i += 1
# 对矩阵每行排序  score1.value  score1.indices
score1 = torch.sort(score, dim=1, descending=True)
# 取所有行 每行前20个坐标
index_top = score1[1][:, 0:20]
# print(index_top[970])  tensor([ 34,  75, 145, 127, 105,  51, 120, 135,  67, 159, 149,  31,  47, 140,
#          64,  93,  36,  57, 134,  80])
# print(index_top.shape) torch.Size([1373, 20])
# show(index_top[633])
# show(index_top[598])
# show(index_top[970])
# 633 Curcumin 18
# 598 Ciprofloxacin 10
# 970 Moxifloxacin
#

fileName = 'top_20_candidates.xlsx'
xw_toExcel(index_top, fileName)
