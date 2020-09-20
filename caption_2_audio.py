#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gtts import gTTS 
import os 
class cap_to_aud():
    
    def _init_(self,text,lang):
        self.text=text
        self.lang=lang
        
    def speak_save(self,text,lang,save_name):
        myobj = gTTS(text=text, lang=lang, slow=False) 
        myobj.save(save_name) 
        #Now if you want to run it
        return os.system("mpg321 alpha.mp3") 


# In[ ]:




