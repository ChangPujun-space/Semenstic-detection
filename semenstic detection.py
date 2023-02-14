#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip3 install snownlp')


# In[1]:


from snownlp import SnowNLP
import pandas as pd
import numpy as np


# In[41]:


textdata = pd.read_csv("D://Users//pjchang//Downloads/XXX.csv",index_col = False)
textdata = textdata[textdata["是否XX范围"] == "T"]
testdata = textdata["commentcontent"]


# In[45]:


print(testdata)
semenstic_scorelist = []
for i in testdata:
    s = SnowNLP(i)
    senmenstic_score = s.sentiments
    print(str(senmenstic_score),i)
    semenstic_scorelist.append(senmenstic_score)


# In[52]:


textdata["情感分析"] = semenstic_scorelist
print(textdata)
textdata.to_csv("D:/Users/pjchang/Desktop/情感分析2.csv",encoding="utf_8_sig")

