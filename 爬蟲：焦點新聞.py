#!/usr/bin/env python
# coding: utf-8

# In[2]:


import requests
from bs4 import BeautifulSoup
html = requests.get("https://tw.yahoo.com/")
s = BeautifulSoup(html.text, "html.parser")
v = s.find_all("a", "story-title")

for s in v:
    print("標題："+s.text)
    print("網址："+"http://twyahoo.com"+s.get("href"))


# In[ ]:




