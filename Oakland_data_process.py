import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


abstract = {}

# 标称属性，生成直方图
def nomial_process(column, figname, year):
    frequency = {}
    for i in column:
        if i not in frequency:
            frequency[i] = 1
        else:
            frequency[i] += 1
    abstract[figname] = frequency
    if len(frequency.keys())<=10:
        plt.bar([str(x) for x in frequency.keys()], [frequency[x] for x in frequency.keys()])
        plt.savefig(figname+year+'.png')
        plt.clf()
    elif len(frequency.keys())<=500:
        plt.bar([str(x) for x in frequency.keys()], [frequency[x] for x in frequency.keys()])
        if 'None' in frequency.keys():
            plt.xticks(['None'])
        else:
            plt.xticks([])
        plt.savefig(figname+year+'.png')
        plt.clf()

# 数值属性，生成盒图
def value_process(column, figname, year):
    ls = []
    sum = 0
    for i in column:
        if i == 'None':
            continue
        i = int(''.join(x for x in i if x.isdigit())[4:-3])
        ls.append(i)
        sum += 1
    ls.sort()
    plt.boxplot(ls)
    plt.savefig(figname+year+'.png')
    plt.clf()
    min = ls[0]
    max = ls[-1]
    if (sum+1) % 4 == 0:
        Q1 = ls[int((sum+1)/4-1)]
        Q3 = ls[int((sum+1)/4*3-1)]
    else:
        Q1 = int((ls[int((sum+1)/4-1)] + ls[int((sum+1)/4)]) / 2)
        Q3 = int((ls[int((sum+1)/4*3-1)] + ls[int((sum+1)/4*3)]) / 2)
    if (sum+1) % 2 == 0:
        Q2 = ls[int((sum+1)/2-1)]
    else:
        Q2 = int((ls[int((sum+1)/2-1)] + ls[int((sum+1)/2)]) / 2)
    min = str(min)[:-8]+'/'+str(min)[-8:-6]+'/'+str(min)[-6:-4]+':'+str(min)[-4:-2]+':'+str(min)[-2:]
    Q1 = str(Q1)[:-8]+'/'+str(Q1)[-8:-6]+'/'+str(Q1)[-6:-4]+':'+str(Q1)[-4:-2]+':'+str(Q1)[-2:]
    Q2 = str(Q2)[:-8]+'/'+str(Q2)[-8:-6]+'/'+str(Q2)[-6:-4]+':'+str(Q2)[-4:-2]+':'+str(Q2)[-2:]
    Q3 = str(Q3)[:-8]+'/'+str(Q3)[-8:-6]+'/'+str(Q3)[-6:-4]+':'+str(Q3)[-4:-2]+':'+str(Q3)[-2:]
    max = str(max)[:-8]+'/'+str(max)[-8:-6]+'/'+str(max)[-6:-4]+':'+str(max)[-4:-2]+':'+str(max)[-2:]
    abstract[figname] = {'min':min, 'Q1':Q1, 'Q2':Q2, 'Q3':Q3, 'max':max}

# 清洗，把空值置为None
def clear_data(data_path):
    data = pd.read_csv(data_path, encoding='utf-8').fillna("None")
    if 'Zip Codes' in data.keys():
        data = data.iloc[:, :-1]
    if 'Location 1' in data.keys():
        data.rename(columns={'Location 1':'Location'},inplace=True)
        for i in range(len(data['Location'])):
            if i % 1000 == 0:
                print(i)
            if data['Location'][i] == 'None':
                continue
            json_dict = eval(data['Location'][i])
            json_dict = eval(json_dict['human_address'])
            data.loc[i, 'Location'] = json_dict['address']
    data.to_csv(data_path, index=False)

# 将缺失部分剔除
def delete_data(data_path):
    data = pd.read_csv(data_path, encoding='utf-8').fillna("None")
    delete = []
    for i in range(len(data)):
        for j in data.keys():
            if data[j][i] == 'None':
                delete.append(i)
                break
    data.drop(delete, inplace=True)
    data.to_csv('delete-'+data_path, index=False)

# 用最高频率值来填补缺失值
def freq_data(data_path):
    data = pd.read_csv(data_path, encoding='utf-8').fillna("None")
    add = {}
    for i in data.keys():
        column = data[i]
        frequency = {}
        max = 0
        for j in column:
            if j not in frequency:
                frequency[j] = 1
            else:
                frequency[j] += 1
            if frequency[j] > max:
                max = frequency[j]
                value = j
        add[i] = value
    for i in range(len(data)):
        for j in data.keys():
            if data[j][i] == 'None':
                data.loc[i, j] = add[j]
    data.to_csv('freq-'+data_path, index=False)


def ent(data):
    prob1 = pd.value_counts(data) / len(data)
    return sum(np.log2(prob1) * prob1 * (-1))

# 通过属性的相关关系来填补缺失值
def relevance_data(data_path):
    data = pd.read_csv(data_path, encoding='utf-8').fillna("None")
    add = {}
    for i in data.keys():
        max = -1
        for j in data.keys():
            if j == i:
                continue
            e1 = data.groupby(i).apply(lambda x:ent(x[j]))
            p1 = pd.value_counts(data[i]) / len(data[i])
            e2 = sum(e1 * p1)
            g = ent(data[j]) - e2
            if g > max:
                max = g
                value = j
        add[i] = value            
    for i in range(len(data)):
        for j in data.keys():
            if data[j][i] == 'None':
                data.loc[i, j] = data[data[add[j]]==data[add[j]][i]][j].value_counts().index[0]
    data.to_csv('rele-'+data_path, index=False)

# 通过数据对象之间的相似性来填补缺失值
def similarity_data(data_path):
    data = pd.read_csv(data_path, encoding='utf-8').fillna("None")
    for i in range(len(data)):
        for j in data.keys():
            if data[j][i] == 'None':
                max = -1
                for k in range(len(data)):
                    if (data.loc[0] == data.loc[1]).sum() > max and data[j][k] != 'None':
                        max = (data.loc[0] == data.loc[1]).sum()
                        value = data[j][k]
                data.loc[i, j] = value
    data.to_csv('simi-'+data_path, index=False)


if __name__ == '__main__':
    Nominal = ['Agency', 'Location', 'Area Id', 'Beat', 'Priority', 'Incident Type Id', 'Incident Type Description']
    Value = ['Create Time', 'Closed Time']
    # 数据文件路径
    data_path = 'freq-records-for-2011.csv'
    year = ''.join(x for x in data_path if x.isdigit())
     # 清洗，把空值置为None
    clear_data(data_path)
    # 补充缺失值的四种方法
    # delete_data(data_path)
    # freq_data(data_path)
    # relevance_data(data_path)
    # similarity_data(data_path)
    data = pd.read_csv(data_path, encoding='utf-8').fillna("None")
    for i in data.keys():
        column = data[i]
        # 对于标称属性，生成直方图
        if i in Nominal:
            nomial_process(column, i, year)
        # 对于数值属性，生成盒图
        elif i in Value:
            value_process(column, i, year)
    with open('abstract'+year+'.json', 'w') as f:
        json.dump(abstract,f)