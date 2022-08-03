# %% [markdown]
# ##### 1. 导入需要的库和数据

# %%
import gensim
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
# from gensim.parsing.preprocessing import preprocess_string,remove_stopwords
import jieba
import pandas as pd
import numpy as np
import random
# import warnings
# warnings.filterwarnings("ignore")

# %%
#读取两张csv表，并进行合并处理
df1 = pd.DataFrame(pd.read_csv('./first sheet v2.csv'))
df2 = pd.DataFrame(pd.read_csv('./second sheet v2.csv'))
df_total = pd.concat([df1,df2],ignore_index=True)
df_practice = df_total[['source', 'content']]

# %% [markdown]
# ##### 2. 获取用来训练的评论数据，逐条进行分停词，然后训练doc2vec模型

# %% [markdown]
# ##### 2.1 获取评论数据

# %%
#仅保留评论数据
def getText():
    discuss_train = list(df_practice['content'])
    return discuss_train
Text = getText()
# Text

# %%
#前两条
print(Text[0:2])

# %% [markdown]
# ##### 2.2 分、停词，进行基本的清洗

# %%
#对评论进行分词、停词操作,为了后期抽样和可视化的便利，本函数每次仅处理一条评论
#这里的停用词表应该和项目之前的有差异，后期需更换停用词表！！！
def cut_sentence(text):
    stop_list = [line.strip() for line in open('stopwords.txt',encoding='UTF-8').readlines()]
    stop_list.append("\n")
    each_cut = jieba.lcut(text)
    each_result = [word for word in each_cut if word not in stop_list]
    each_result_str =" ".join(str(i) for i in each_result) 
    return each_result_str

# %%
#给定一个用来训练句向量的序号列表，可以用来选择使用的训练语料
#前500条
train_num = [item for item in range(67248)]

# %%
#对用来训练的评论，逐句进行分、停词
Text_Seg = list()#将清洗完的评论，以字符串形式，放入列表储存
for item in train_num:
    Text_Seg.append(cut_sentence(Text[item]))


# %%
# 展示清洗后的数据
print(Text_Seg[0:2])

# %% [markdown]
# ##### 2.3 将评论转化为符合训练要求的格式

# %%
#将句子转化为符合gensim.models.Doc2Vec库要求的格式
TaggededDocument = gensim.models.doc2vec.TaggedDocument
def X_train(seg_Text):
    x_train = []
    for i,text in enumerate(seg_Text):
        word_list = text.split(' ')
        length = len(word_list)
        word_list[length-1]=word_list[length-1].strip()
        document = TaggededDocument(word_list,tags=[i])
        x_train.append(document)
    return x_train

#Text_pred保存了格式规范化后的训练数据
Text_pred = X_train(Text_Seg)

# %%
#展示处理后的格式
print(Text_pred[0:2])

# %% [markdown]
# ##### 2.4 训练模型

# %%
#doc2vec正式训练
model = gensim.models.Doc2Vec(vector_size=300, window=10, min_count=5,
                            workers=4, alpha=0.025, min_alpha=0.025, epochs=12)
model.build_vocab(Text_pred)
print("开始训练...")
# 训练模型
model.train(Text_pred, total_examples=model.corpus_count, epochs=12)
model.save("doc2vec.model")
print("模型已保存")

# %% [markdown]
# ##### 2.5输出对应向量

# %%
#根据训练得到的模型，将文本转化为向量
def sent2vec(model, words):
    vect_list = []
    for w in words:
        try:
            vect_list.append(model.wv[w])
        except:
            continue
    vect_list = np.array(vect_list)
    vect = vect_list.sum(axis=0)
    return vect / np.sqrt((vect ** 2).sum()) 

# %%
#将句向量结果用Dataframe格式保存
doc_num2vec_dict = dict() #创建一个序号对应向量的字典
for item in range(67248): 
    doc_num2vec_dict[item] = sent2vec(model,Text[item])
    
matrix = pd.DataFrame() #用Dataframe格式保存
for i in range(67248):
    try:
        array_i = pd.DataFrame(sent2vec(model,Text[i])).T
        matrix = pd.concat([matrix,array_i],axis=0,ignore_index=True)
    except:
        array_i = pd.DataFrame(np.zeros(300)).T
        matrix = pd.concat([matrix,array_i],axis=0,ignore_index=True)

# %%
matrix.to_csv("matrix.csv")

# %% [markdown]
# ##### 3.降维与可视化分析

# %% [markdown]
# ##### 3.1PCA降维

# %%
#PCA降维
from sklearn.decomposition import PCA
pca = PCA(n_components=2) 
pca = pca.fit(matrix)  # 拟合模型
matrix_ld = pd.DataFrame(pca.transform(matrix))  # 获取降维后的新矩阵 #low dimensions

# %% [markdown]
# ##### 3.2对matrix_ld矩阵补充平台来源属性

# %%
#获取评论对应的平台编码
df_total = pd.concat([df1,df2],ignore_index=True)
label = pd.DataFrame(df_total)
subtotal = label.groupby(['source']).count()
print(subtotal)
#平台来源编码
label.loc[label['source']=="微博",['source']] = 0 
label.loc[label['source']=="知乎问答",['source']] = 1
label.loc[label['source']=="简书文章",['source']] = 2 
label.loc[label['source']=="豆瓣日记",['source']] = 3 

def getSource():
    label_targets = label['source'] #label_targets用来表示平台来源
    label_targets = pd.DataFrame(label_targets)
    return pd.DataFrame(label_targets)
# 加入source的编码
matrix_visualization = pd.concat([matrix_ld,getSource()],axis=1,ignore_index=False)

# %%
matrix_visualization.to_csv("matrix_visualization.csv")

# %% [markdown]
# ##### 3.3可视化

# %%
#可视化
import seaborn as sns
import matplotlib.pyplot as plt

# %%
import numpy as np
import matplotlib.pyplot as plt  #导入绘图操作用到的库
import random

#读取数据得到一个字典类型数据，需要根据键名 ’data‘ 取出对应的值。
data  = matrix_visualization
fig = plt.figure(dpi=1000)  # 创建画布
ax = fig.add_subplot(111)


idx_0 = data[data['source'] == 0].index.to_list()  # 找出标签为0的样本行数
ns_0 = round(len(idx_0)/73)
samples_0=random.sample(idx_0,ns_0)#随机抽取1/10

idx_1 = data[data['source'] == 1].index.to_list()  # 找出标签为1的样本行数
ns_1 = round(len(idx_1)/10)
samples_1=random.sample(idx_1,ns_1)#随机抽取1/10

idx_2 = data[data['source'] == 2].index.to_list()  # 找出标签为2的样本行数
ns_2 = round(len(idx_2)/10)
samples_2=random.sample(idx_2,ns_2)#随机抽取1/10

idx_3 = data[data['source'] == 3].index.to_list()  # 找出标签为3的样本行数
ns_3 = round(len(idx_3)/3)
samples_3=random.sample(idx_3,ns_3)#随机抽取1/10

#绘制散点图
p0 = ax.scatter(data.loc[samples_0,'PCA1'], data.loc[samples_0,'PCA2'], marker='.', color='r', s=0.5,label = 'Weibo')
p1 = ax.scatter(data.loc[samples_1,'PCA1'], data.loc[samples_1,'PCA2'], marker='.', color='g', s=0.5,label = 'Zhihu')
p2 = ax.scatter(data.loc[samples_2,'PCA1'], data.loc[samples_2,'PCA2'], marker='.', color='y', s=0.5,label = 'Jianshu')
p3 = ax.scatter(data.loc[samples_3,'PCA1'], data.loc[samples_3,'PCA2'], marker='.', color='b', s=0.5,label = 'Douban')

ax.legend() 
plt.show()   #显示散点图

# %%


# %%



