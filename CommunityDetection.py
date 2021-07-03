#!/usr/bin/env python
# coding: utf-8

# In[1]:


import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark import SparkContext, SparkConf
from graphframes import GraphFrame
import time


# In[2]:


#初始化并读取所有数据
#conf = SparkConf().setAppName("MySpark").setMaster("local[*]")
#sc = SparkContext(conf=conf)
#lines = sc.textFile('twitter_combined.txt')
spark = SparkSession.builder.appName("MySpark").getOrCreate()
sc = spark.sparkContext
lines = sc.textFile('twitter_sub.txt')


# In[3]:


#提取所有实体并去重，然后转换成dataFrame
entity = lines.flatMap(lambda x:x.split(' ')).distinct()
entity_zip = entity.zipWithIndex()
entity_row = entity_zip .map(lambda e: Row(id = e[1], name = e[0]))
entity_table = spark.createDataFrame(entity_row) 
entity_table.show()


# In[4]:


#实体数量
entity.count()


# In[5]:


#提取所有关系，然后转换成dataFrame（实体id，实体id，关系）
entity_dict = dict(entity_zip .map(lambda e: (e[0], e[1])).collect())
relation_row = lines.map(lambda x:x.split(' ')).map(lambda r: Row(src = entity_dict[r[0]], dst = entity_dict[r[1]],                                                                  relationship = 'concerned '))
relation_table = spark.createDataFrame(relation_row) 
relation_table.show()


# In[6]:


#关系数量
relation_row.count()


# In[7]:


#将提取出来的实体表和关系表写入csv文件夹，以供查看
entity_table.toPandas().to_csv('entity2id.csv',encoding='utf_8_sig',index=False)
relation_table.toPandas().to_csv('relation.csv',encoding='utf_8_sig',index=False)


# In[8]:


#构建Graph
g = GraphFrame(entity_table, relation_table)


# In[9]:


#入度取前10展示
g.inDegrees.sort("inDegree",ascending=False).show(10)


# In[10]:


#使用PageRank算法选出重要用户
tic = time.time()
result = g.pageRank(resetProbability=0.15, maxIter=5)
result.vertices.sort('pagerank',ascending=False).show(10)
toc = time.time()
print("PageRank算法用时:" + str(toc-tic))


# In[16]:


#使用LPA算法来做社区发现（CommunityDetection）
tic = time.time()
results = g.labelPropagation(maxIter=5)
results.orderBy('label').show(10)
toc = time.time()
print("LPA算法用时:" + str(toc-tic))


# In[18]:


#将结果写入csv文件夹，以供查看
results.toPandas().to_csv('results.csv',encoding='utf_8_sig',index=False)


# In[26]:


#查看社区分类情况
results.groupBy('label').count().collect()


# In[27]:


sc.stop()

