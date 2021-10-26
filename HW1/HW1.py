#!/usr/bin/env python
# coding: utf-8

# # 2021 Intro. to Machine Learning 
# ## Program Assignment #1 - Naïve Bayes
# 
# 
# ### 0816153 陳琮方

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.interpolate import interp1d
warnings.simplefilter(action="ignore", category=FutureWarning)


# ## Data Input
# 
# ##### 使用 pandas 讀取 dataset，並觀察 features 以及 dataset 大小

# In[2]:


iris_data_path = 'iris.data'
mush_data_path = 'agaricus-lepiota.data'


# In[3]:


mush_header = [
    "edible=e, poisonous=p", # Target
    "cap-shape", "cap-surface", "cap-color", "bruises?",
    "odor", "gill-attachment", "gill-spacing", "gill-size", "gill-color",
    "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
    "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
    "ring-number", "ring-type", "spore-print-color", "population", "habitat"
]


# In[4]:


iris_header = [
    "sepal length", "sepal width",
    "petal length", "petal width",
    "class" # Target
]


# In[5]:


iris_data = pd.read_csv(iris_data_path, header = None)
mush_data = pd.read_csv(mush_data_path, header = None)

iris_data.columns = iris_header
mush_data.columns = mush_header

print(f'Iris data shape = {iris_data.shape}')
print(f'Mush data shape = {mush_data.shape}')


# ## Data Visualization

# ### Mushroom
# * Show the data distribution by value frequency of every feature.
# * Split data based on their labels (targets) and show the data distribution of each feature again.
# 
# ##### 使用 matplotlib 搭配 seaborn 中的 bar plot ，畫出各項 features 不同類型的數量
# ##### 以及將 Data 用 Label 分類，在觀察各項 features 的分佈

# In[6]:


num_features = mush_data.shape[1]

num_y = 4
num_x = (num_features + 3) // num_y

sns.set(font_scale = 2)
fig, axes = plt.subplots(num_x, num_y, sharex = False, figsize=(40, 64))
now_x = 0
now_y = 0

for i in range(1, num_features):
    now_data = mush_data[mush_header[i]].tolist()
    fet_data = [j for j in set(now_data)]
    cnt_data = [now_data.count(j) for j in fet_data ]
    
    now_df = pd.DataFrame({'feature' : fet_data, 'count' : cnt_data})
    now_df = now_df.sort_values("count",ascending=False)
    
    p = sns.barplot(x = 'feature', y = 'count', data = now_df, ax = axes[now_x][now_y])
    p.set(xlabel = mush_header[i], ylabel = "count")
    now_y += 1
    if now_y == num_y:
        now_y = 0
        now_x += 1
        
plt.savefig('Mushroom_features_distribution.png')


# In[7]:


num_features = mush_data.shape[1]

num_y = 4
num_x = (num_features + 3) // num_y

fig, axes = plt.subplots(num_x, num_y, sharex = False, figsize=(40, 64))
now_x = 0
now_y = 0

for i in range(1, num_features):
    now_data = mush_data[[mush_header[0], mush_header[i]]]
    p = sns.countplot(
            hue = mush_header[0], 
            x = mush_header[i], 
            data = now_data, 
            ax = axes[now_x][now_y], palette = "Set2"
    )
    p.set(xlabel = mush_header[i], ylabel = "count")
    now_y += 1
    if now_y == num_y:
        now_y = 0
        now_x += 1

        
plt.savefig('Mushroom_target_feature.png')


# ###  Iris
# * Show the data distribution by average, standard deviation, and value         frequency(binning might be needed) of every feature.
# * Split data based on their labels (targets) and show the data distribution of each feature again. 
# 
# ##### 這裡我使用 seaborn 中的 distplot 以及 boxplot 去看 iris dataset 中各個 feature 的數據分佈狀況
# ##### 增加 histplot 看不同類型之間的分佈狀況

# In[8]:


num_features = iris_data.shape[1]
now_linewidt = 10

sns.set(font_scale = 2)
fig, axes = plt.subplots(num_features, sharex = False, figsize=(20, 50))

for i in range(num_features - 1):
    now_data = iris_data[iris_header[i]]
    
    p = sns.distplot(x = now_data, ax = axes[i], bins = 10)
    p.set(xlabel = iris_header[i], ylabel = "cm")
    
    aver = np.average(now_data)
    sigma = np.std(now_data)

    print(f"{iris_header[i]} 𝜇  = {aver}, 𝜎 = {sigma}")
    
    low = aver - sigma
    hi = aver + sigma
    
    line = p.lines[0].get_data()
    ipf = interp1d(x=line[0], y=line[1])

    p.plot([low, low], [0, ipf(low)], linewidth = now_linewidt)
    p.plot([hi, hi], [0, ipf(hi)], linewidth = now_linewidt)
    p.plot([aver, aver], [0, ipf(aver)], linewidth = now_linewidt)

sns.boxplot(data = iris_data, ax = axes[num_features - 1])
plt.savefig('Iris_features_distribution.png')


# In[9]:


num_features = iris_data.shape[1]

sns.set(font_scale = 2)
fig, axes = plt.subplots(num_features - 1, sharex = False, figsize=(20, 50))

for i in range(num_features - 1):
    now_data = iris_data[[iris_header[i], 'class']]
    
    p = sns.histplot(
        x = iris_header[i], 
        hue = "class", 
        data = now_data, 
        ax = axes[i], 
        bins = 10
    )
    p.set(xlabel = iris_header[i], ylabel = "cm")

plt.savefig('Iris_target_feature.png')


# In[10]:


iris_class = list(set(iris_data[iris_header[-1]].tolist()))

num_class = len(iris_class)

for j in range(num_class):
    now_data = iris_data.loc[iris_data[iris_header[-1]] == iris_class[j]]
    
    print(iris_class[j])
    
    num_features = now_data.shape[1]
    
    sns.set(font_scale = 2)
    fig, axes = plt.subplots(num_features - 1, sharex = False, figsize=(20, 50))

    plt.title(f'{iris_class[j]}')
    for i in range(num_features - 1):
        tmp_data = now_data[iris_header[i]]
        
        p = sns.distplot(x = tmp_data, ax = axes[i], bins = 10)
        p.set(xlabel = iris_header[i], ylabel = "cm")

        aver = np.average(tmp_data)
        sigma = np.std(tmp_data)

        print(f"{iris_header[i]} 𝜇  = {aver}, 𝜎 = {sigma}")

        low = aver - sigma
        hi = aver + sigma

        line = p.lines[0].get_data()
        ipf = interp1d(x=line[0], y=line[1])

        p.plot([low, low], [0, ipf(low)], linewidth = now_linewidt)
        p.plot([hi, hi], [0, ipf(hi)], linewidth = now_linewidt)
        p.plot([aver, aver], [0, ipf(aver)], linewidth = now_linewidt)


# ## Data Preprocessing
# * Drop features with any missing value.
# * Transform data format and shape so your model can process them.
# * Shuffle the data. (Do it at Model Construction)
# 
# 
# #### **Data origin size**
# ```
# Iris data shape = (150, 5)
# Mush data shape = (8124, 23)
# ```
# #### **Data after drop**
# ```
# Iris data shape = (150, 5)
# Mush data shape = (8124, 22)
# ```

# In[11]:


iris_data_drop = iris_data.replace(['', ' ', '?'], np.nan)
mush_data_drop = mush_data.replace(['', ' ', '?'], np.nan)

iris_data_drop = iris_data_drop.dropna(axis = 'columns')
mush_data_drop = mush_data_drop.dropna(axis = 'columns')

print(f'Iris data after drop shape = {iris_data_drop.shape}')
print(f'Mush data after drop shape = {mush_data_drop.shape}')


# In[12]:


from sklearn.preprocessing import OrdinalEncoder

mush_x = mush_data_drop.drop(mush_header[0] , axis = 1)
mush_y = mush_data_drop[mush_header[0]]
mush_label = list(set(mush_y.tolist()))

iris_x = iris_data_drop.drop(iris_header[-1], axis = 1)
iris_y = iris_data_drop[iris_header[-1]]
iris_label = list(set(iris_y.tolist()))

enc = OrdinalEncoder()
enc.fit(mush_x)
mush_x = enc.transform(mush_x)

iris_x = iris_x.to_numpy()
iris_y = iris_y.to_numpy()


# ## Model Construction
# 
# ### 本次作業都使用 sklearn package 所提供 model
# 
# #### 在 ```model_selection.train_test_split``` 中，將 ```test_size``` 設為 0.30 即為將 Holdout validation ratio 設為 7:3
# #### 而在 ```model_selection.KFold``` 中，```n_splits``` 參數為 K-fold 的次數

# In[13]:


from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB
from sklearn import model_selection
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, precision_score
from sklearn.metrics import classification_report

from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action="ignore", category=ConvergenceWarning)


# In[14]:


def print_heatmap(mat, title):
    sns.heatmap(mat, square= True, annot=True, cbar= True)
    plt.xlabel("Predicted value")
    plt.ylabel("True value")
    plt.title(f"{title}")
    plt.show()


# In[15]:


def get_Result_Dic(y_test, y_predict, now_label, title, draw = True) -> dict:
    ret = {}
    
    ret['accuracy'] = accuracy_score(y_test, y_predict)
    ret['mat'] = pd.DataFrame(
        confusion_matrix(y_test, y_predict, labels = now_label),
        index = now_label,
        columns = now_label
    )
    ret['recall'] = recall_score(y_test, y_predict, average=None, labels = now_label)
    ret['precision'] = precision_score(y_test, y_predict, average=None, labels = now_label)
    ret['report'] = classification_report(
        y_test, y_predict, target_names = now_label
    )
    if draw == False:
        return ret 
    
    print_heatmap(ret['mat'], title)
    
    return ret


# ### Mushroom

# In[16]:


mush_result = {}


# In[17]:


def mush_holdout_lap(now_alpha = 1e3, typ = 0):
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
            mush_x, mush_y, test_size=0.30, shuffle=True, random_state=50)
    
    if typ == 0:
        clf = MultinomialNB(alpha = now_alpha)
    else:
        clf = CategoricalNB(alpha = now_alpha)
        
    clf.fit(X_train, Y_train)
    y_predict = clf.predict(X_test)
    
    ret = get_Result_Dic(
        Y_test, y_predict, mush_label, f"Mushroom holdout alpha = {now_alpha}")
    
    if typ == 0:
        mush_result[f'MultinomialNB, Holdout alpha = {now_alpha}'] = ret
    else:
        mush_result[f'CategoricalNB, Holdout alpha = {now_alpha}'] = ret
        
    print(ret['report'])


# In[18]:


#  Holdout with Laplace smoothing
print("Sklearn MultinomialNB")
mush_holdout_lap()

print("Sklearn CategoricalNB")
mush_holdout_lap(typ = 1)


# ### Mushroom Holdout Result (Alpha = 1000)
# 
# **Sklearn MultinomialNB**
# ```
# 
#               precision    recall  f1-score   support
# 
#            p       0.72      0.89      0.79      1219
#            e       0.85      0.65      0.74      1219
# 
#     accuracy                           0.77      2438
#    macro avg       0.78      0.77      0.77      2438
# weighted avg       0.78      0.77      0.77      2438
# ```
# 
# **Sklearn CategoricalNB**
# ```
# 
#               precision    recall  f1-score   support
# 
#            p       0.82      0.98      0.89      1219
#            e       0.98      0.79      0.87      1219
# 
#     accuracy                           0.88      2438
#    macro avg       0.90      0.88      0.88      2438
# weighted avg       0.90      0.88      0.88      2438
# ```

# In[19]:


#  Holdout without Laplace smoothing (alpha = 1e-9)
print("Sklearn MultinomialNB")
mush_holdout_lap(1e-9)

print("Sklearn CategoricalNB")
mush_holdout_lap(1e-9, typ = 1)


# ### Mushroom Holdout Result (Alpha = 1e-9)
# 
# **Sklearn MultinomialNB**
# 
# ```
#               precision    recall  f1-score   support
# 
#            p       0.75      0.94      0.83      1219
#            e       0.92      0.68      0.78      1219
# 
#     accuracy                           0.81      2438
#    macro avg       0.83      0.81      0.81      2438
# weighted avg       0.83      0.81      0.81      2438
# ```
# 
# **Sklearn CategoricalNB**
# ```
#               precision    recall  f1-score   support
# 
#            p       1.00      0.99      1.00      1219
#            e       0.99      1.00      1.00      1219
# 
#     accuracy                           1.00      2438
#    macro avg       1.00      1.00      1.00      2438
# weighted avg       1.00      1.00      1.00      2438
# ```
# 

# In[20]:


def mush_kfold_lap(now_alpha = 1e3, typ = 0):
    kfold = model_selection.KFold(n_splits=3, shuffle=True, random_state=50)
    kfold.get_n_splits(mush_x)
    
    flag = 0
    rec_sum = None
    acc_sum = None
    mat_sum = None
    pre_sum = None
    for train_index, test_index in kfold.split(mush_x):
        if typ == 0:
            clf = MultinomialNB(alpha = now_alpha)
        else:
            clf = CategoricalNB(alpha = now_alpha)
            
        clf.fit(mush_x[train_index], mush_y[train_index])
        y_predict = clf.predict(mush_x[test_index])

        ret = get_Result_Dic(
            mush_y[test_index], y_predict, mush_label, f"Mushroom alpha = {now_alpha}",
            draw = False
        )
        if typ == 0:
            mush_result[f'MultinomialNB, K-fold alpha = {now_alpha}'] = ret
        else:
            mush_result[f'CategoricalNB, K-fold alpha = {now_alpha}'] = ret
            
        if flag != 0: 
            flag += 1
            mat_sum += ret['mat']
            rec_sum += ret['recall']
            acc_sum += ret['accuracy']
            pre_sum += ret['precision']
        else:
            flag = 1
            mat_sum = ret['mat']
            rec_sum = ret['recall']
            acc_sum = ret['accuracy']
            pre_sum = ret['precision']
    
    print(f'Accuracy  ave. = {acc_sum / flag}')
    print(f'Recall    ave. = {rec_sum / flag}')
    print(f'Precision ave. = {pre_sum / flag}')
    print(mush_label)
    mat_sum /= flag
    print_heatmap(mat_sum, f'Mushroom kfold sum, alpha = {now_alpha}')
            


# In[21]:


#  k-fold with Laplace smoothing (alpha = 1e3)
print("Sklearn MultinomialNB")
mush_kfold_lap()

print("Sklearn CategoricalNB")
mush_kfold_lap(typ = 1)


# ### Mushroom K-fold Result (Alpha = 1000)
# 
# **Sklearn MultinomialNB**
# ```
# Accuracy  ave. = 0.774741506646972
# Recall    ave. = [0.65355775 0.88782751]
# Precision ave. = [0.84378999 0.73339463]
# ['p', 'e']
# ```
# 
# **Sklearn CategoricalNB**
# ```
# Accuracy ave. = 0.8916789758739538
# Recall ave. = [0.79216559 0.98425852]
# Precision ave. = [0.97919356 0.83550931]
# ['p', 'e']
# ```

# In[22]:


#  k-fold without Laplace smoothing (alpha = 1e-9)
print("Sklearn MultinomialNB")
mush_kfold_lap(1e-9, 0)

print("Sklearn CategoricalNB")
mush_kfold_lap(1e-9, 1)


# ### Mushroom K-fold Result (Alpha = 1e-9)
# 
# **Sklearn MultinomialNB**
# ```
# Accuracy  ave. = 0.8136386016740521
# Recall    ave. = [0.67966638 0.93829421]
# Precision ave. = [0.91077083 0.75876575]
# ['p', 'e']
# ```
# **Sklearn CategoricalNB**
# ```
# Accuracy  ave. = 0.9969226981782372
# Recall    ave. = [0.99923847 0.99470085]
# Precision ave. = [0.99447597 0.99927866]
# ['p', 'e']
# ```

# ### Iris

# In[23]:


iris_result = {}


# In[24]:


def iris_holdout():
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
            iris_x, iris_y, test_size=0.30, shuffle=True, random_state=50)
    
    clf = GaussianNB()
    clf.fit(X_train, Y_train)
    
    y_predict = clf.predict(X_test)
    
    ret = get_Result_Dic(Y_test, y_predict, iris_label, f"Iris holdout")
    iris_result[f'Holdout'] = ret
                         
    print(ret['report'])


# In[25]:


iris_holdout()


# ### Iris Holdout Result 
# ```
#                  precision    recall  f1-score   support
# 
# Iris-versicolor       1.00      1.00      1.00        14
#  Iris-virginica       0.94      0.94      0.94        17
#     Iris-setosa       0.93      0.93      0.93        14
# 
#        accuracy                           0.96        45
#       macro avg       0.96      0.96      0.96        45
#    weighted avg       0.96      0.96      0.96        45
# ```

# In[26]:


def iris_kfold():
    kfold = model_selection.KFold(n_splits=3, shuffle=True, random_state=50)
    kfold.get_n_splits(iris_x)
    
    flag = 0
    rec_sum = None
    acc_sum = None
    mat_sum = None
    pre_sum = None
    for train_index, test_index in kfold.split(iris_x):
        clf = GaussianNB()
        
        clf.fit(iris_x[train_index], iris_y[train_index])
        y_predict = clf.predict(iris_x[test_index])

        ret = get_Result_Dic(
            iris_y[test_index], y_predict, iris_label, f"Iris",
            draw = False
        )
        iris_result[f'K-fold'] = ret
        if flag != 0: 
            flag += 1
            mat_sum += ret['mat']
            rec_sum += ret['recall']
            acc_sum += ret['accuracy']
            pre_sum += ret['precision']
        else:
            flag = 1
            mat_sum = ret['mat']
            rec_sum = ret['recall']
            acc_sum = ret['accuracy']
            pre_sum = ret['precision']
    
    print(f'Accuracy  ave. = {acc_sum / flag}')
    print(f'Recall    ave. = {rec_sum / flag}')
    print(f'Precision ave. = {pre_sum / flag}')
    print(iris_label)
    
    mat_sum /= flag
    print_heatmap(mat_sum, f'Iris kfold ave.')


# In[27]:


iris_kfold()


# ### Iris K-fold Result
# ```
# Accuracy  ave. = 0.9533333333333333
# Recall    ave. = [0.93995098 0.92916667 1.        ]
# Precision ave. = [0.9248366  0.93842593 1.        ]
# ['Iris-versicolor', 'Iris-virginica', 'Iris-setosa']
# ```

# ## Results Comparison & Conclusion

# ### Mushroom
# 
# 從下表中，可以看到不管是用 ```MultinomialNB``` or ```CategoricalNB```，當加上了 Laplace 都造成了 accuracy 下降，以及 racall 跟 precision 。但這個結論不能推論到全部的 Dataset ，這次為了加強差異才將 alpha 設大一點，以及造成影響也不一定都是不好的，要看 trainning 時使用的數據，以及 data 的特性。

# In[28]:


df = pd.DataFrame.from_dict(mush_result, orient = 'index')
df = df.drop('report', axis = 1)
df = df.drop('mat', axis = 1)
df


# ### Iris
# 在下表中可以發現， Holdout 的 accuracy 比 K-fold 還要高，但也跟之中的 shuffle 有關係， training set 不一樣，也會有不一樣的結果，所以一個好的模型中，需要非常多的實驗與測試，選擇最好的那個方法。

# In[29]:


df = pd.DataFrame.from_dict(iris_result, orient = 'index')
df = df.drop('report', axis = 1)
df = df.drop('mat', axis = 1)
df


# ## Questions

# #### Q1. 𝑃(𝑋𝑠𝑡𝑎𝑙𝑘−𝑐𝑜𝑙𝑜𝑟−𝑏𝑒𝑙𝑜𝑤−𝑟𝑖𝑛𝑔|𝑌=𝑒) with and without Laplace smoothing by bar charts 
# 
# 重下面的 distrubution 可以看到，當加入的 Laplace ，會使 0 的機率上升，造成 accuracy 下降，但也不一定每次都是下降，加入 Laplace 這樣可以避免某些類別機率為 0 的狀況（可能是 trainning set 沒有相關數據的關係）。

# In[30]:


now_data = mush_data.loc[mush_data[mush_header[0]] == 'e']
tar_tags = 'stalk-color-below-ring'

col_list = list(set(mush_data[tar_tags].tolist()))
num_feat = len(col_list)
num_alle = now_data.shape[0]

dic = {}
for i in col_list:
    tmp = now_data.loc[now_data[tar_tags] == i]
    dic[i] = tmp.shape[0]

LAP_K = 1000
words = []
couts = []
count_lap = []

for i in dic:
    words.append(i)
    couts.append(dic[i] / num_alle)
    count_lap.append((dic[i] + LAP_K) / (num_alle + LAP_K * num_feat))
    
df = pd.DataFrame({ "feature" : words, "P" : couts, "P_lap" : count_lap})
df = df.sort_values("P",ascending=False)

sns.set(font_scale = 2, rc = {'figure.figsize':(11.7,8.27)})
fig, axes = plt.subplots(1, 2, sharex = False, figsize=(20, 10))


p = sns.barplot(x = "feature", y = "P", data = df, ax = axes[0])
p.set(title = "P(color | e)  without Laplace")

p = sns.barplot(x = "feature", y = "P_lap", data = df, ax = axes[1])
p.set(title = "P(color | e)  with Laplace = 1000")


# #### Q2. What are the values of 𝜇 and 𝜎 of assumed 𝑃(𝑋𝑝𝑒𝑡𝑎𝑙_𝑙𝑒𝑛𝑔𝑡ℎ|𝑌=Iris Versicolour)? 
# #### Q3. Use a graph to show the probability density function of assumed 𝑃(𝑋𝑝𝑒𝑡𝑎𝑙_𝑙𝑒𝑛𝑔𝑡ℎ|𝑌=Iris Versicolour)

# In[31]:


now_data = iris_data.loc[iris_data[iris_header[-1]] == 'Iris-versicolor']

tar_tags = 'petal length'
now_data = now_data[tar_tags]
now_data = now_data.tolist()

p = sns.distplot(x = now_data, bins = 10)
p.set(xlabel = tar_tags, ylabel = "cm")

aver = np.average(now_data)
sigma1 = np.std(now_data, ddof = 0)
sigma2 = np.std(now_data, ddof = 1)

low = aver - sigma1
hi = aver + sigma1

line = p.lines[0].get_data()
ipf = interp1d(x=line[0], y=line[1])

p.plot([low, low], [0, ipf(low)], linewidth = now_linewidt)
p.plot([hi, hi], [0, ipf(hi)], linewidth = now_linewidt)
p.plot([aver, aver], [0, ipf(aver)], linewidth = now_linewidt)
p.set(title = "P(X_{petal length}|Y=Iris-versicolor)")

print(f"𝜇  = {aver}, 𝜎 = {sigma1} (ddof = 0)")
print(f"𝜇  = {aver}, 𝜎 = {sigma2} (ddof = 1)")


# #### Output
# ```
# 𝜇  = 4.26, 𝜎 = 0.4651881339845203  (ddof = 0)
# 𝜇  = 4.26, 𝜎 = 0.46991097723995795 (ddof = 1)
# ```
