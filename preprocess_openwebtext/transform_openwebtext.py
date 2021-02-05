import nltk
import pandas as pd
import re
import numpy as np
import time
import os

from datasets import load_dataset
from tqdm import tqdm
import json
import csv

from nltk.corpus import stopwords
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import word_tokenize, sent_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

import multiprocessing
from joblib import Parallel, delayed
import joblib
from collections import Counter
from scipy.sparse import csr_matrix

#nltk.download('punkt') #à executer une fois pour charger les données.
#nltk.download('stopwords') # à executer une fois pour charger les données.

english_stopwords=set(stopwords.words("english"))
detokenizer=TreebankWordDetokenizer()

def clean(x):
    x= x.lower()
    x= re.sub("[^ \w]"," ",x)
    x=re.sub("(\s\d+\s|^\d+\s)", " ", x)
    x=re.sub(" \d+", " <NUM> ", x) 
    x= re.sub("  "," ",x)
    words= word_tokenize(x)
    words = [w for w in words if not w in english_stopwords]
    clean_x = detokenizer.detokenize(words)
    return clean_x

class FruitFlyVectorizer(BaseEstimator, TransformerMixin):
    
    def __init__(self, max_words = 200,window=6 ):
        self.max_words = max_words
        self.window=window
        self.vect_dictionnary={}
        self.freq_dictionnary={}
        self.columns=[]
    
    def __getstate__(self):
        """Renvoie le dictionnaire d'attributs à sérialiser"""
        dict_attr = dict(self.__dict__)
        return dict_attr
    
    def __setstate__(self, dict_attr):
        """Méthode appelée lors de la désérialisation de l'objet"""
        self.__dict__ = dict_attr
        
    def get_vect_dictionnary(self):
        return self.vect_dictionnary()
    
    def get_freq_dictionnary(self):
        return self.freq_dictionnary()
    
    def parallel_fit(self, dataset, path,file_name):
        num_cores=1# multiprocessing.cpu_count() -1
        data_length= 1000#len(dataset)
        data_part_length= 125 #1000000
        dict_freq = {}
        plus_one = 1 if data_length%data_part_length != 0 else 0
        for i in range(int(data_length/data_part_length)+plus_one):
            # Fit par batch de taille data_part_length
            print("million == "+str(i+1))
            inf= i*data_part_length 
            sup = min(data_length, (i+1)*data_part_length)
            length = sup-inf+1
            step = int(length / num_cores) + 1
            processed_list = Parallel(n_jobs=num_cores)(delayed(self.fit)(dataset[inf+step*i:inf+min(step*(i+1),length)]) for i in range(0,num_cores))
            for sub_dic in processed_list:
                dict_freq = dict(Counter(sub_dic)+Counter(dict_freq))
            

        dict_freq_ordered={k: v for k, v in sorted(dict_freq.items(), key=lambda item: item[1], reverse=True)}
        self.freq_dictionnary= {k:v for i,(k,v) in enumerate(dict_freq_ordered.items()) if i<self.max_words}
        
        dict_freq_ordered={k:i for i,(k,v) in enumerate(dict_freq_ordered.items()) if i<self.max_words}
        self.vect_dictionnary=dict_freq_ordered
        
        self.max_words=len(dict_freq_ordered)
        self.columns= columns= [ 'context_' + i for i in self.vect_dictionnary.keys()]+["target_"+j for j in self.vect_dictionnary.keys()]
        
        f = open(path+"/dict_freq_ordered_"+ file_name + ".json", "w")
        dumped = json.dumps(dict_freq_ordered)
        f.write(dumped)
        f.close()

        f = open(path+"/freq_dictionnary_"+ file_name + ".json", "w")
        dumped = json.dumps(self.freq_dictionnary)
        f.write(dumped)
        f.close()
        
        return self
    
    def fit(self, dataset):
        """apprendre les paramètres necessaires à l'extraction de features"""
        
        dictionnary={}
        cpt=0
        for x in dataset["text"]:
            sents= sent_tokenize(x)
            for s in sents:
                clean_sent= clean(s)
                cpt+=1
                words=word_tokenize(clean_sent)
                for word in words:
                    if word in dictionnary.keys():
                        dictionnary[word]+=1
                    else:
                        dictionnary[word]=1
        return dictionnary
    
    def tokenize(self,words):
        vect=np.zeros(self.max_words, dtype=bool)
        for word in words:
            if word in self.vect_dictionnary.keys():
                i=self.vect_dictionnary[word]
                vect[i]=True
        return vect
    
    def parallel_transform(self, dataset, path, file_name):
        num_cores=multiprocessing.cpu_count() -1
        data_length= len(dataset)
        data_part_length= 10000
        plus_one = 1 if data_length%data_part_length != 0 else 0
        dict_freq = {}
        # Transform par batch de taille data_part_length
        for i in range(int(data_length/data_part_length)+plus_one):
            print("million 0.5 * == "+str(i))
            inf= i*data_part_length 
            sup = min(data_length, (i+1)*data_part_length)
            length = sup-inf+1
            step = int(length / num_cores) + 1
            print("Multiprocessing")
            processed_vects=Parallel(n_jobs=num_cores)(delayed(self.transform)(dataset[inf+step*i:inf+min(step*(i+1),length)], path, file_name) for i in range(0,num_cores))
            
            flatten_vect=[]
            for part_vect in processed_vects:
                flatten_vect+=part_vect
            with open(path + '/vectorized_data_'+file_name + "_"+str(i)+ '.csv', "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(flatten_vect)
            processed_vects = None
            
        
            

    def transform(self, dataset,path, file_name):
        """extraire les features à partir des données en entrée """
        vect=[]
        cpt=0
        pid=os.getpid()
        num_file=0

        for value in tqdm(dataset["text"]):
            sents= sent_tokenize(value)
            for s in sents:
                sent= clean(s)
                words=word_tokenize(sent)
                rng=len(words)-self.window+1
                if rng>0:
                    start_time=time.time()
                    for i in range(rng):
                        subwords=words[i:i+self.window]
                        index= i + int(self.window/2)
                        target= subwords[index:index+1]
                        context= subwords[:index]+subwords[index+1:]
                        vect_target= self.tokenize(target)
                        vect_context= self.tokenize(context)
                        vect_combined= np.hstack([vect_context,vect_target])
                        cpt+=1
                        vect.append(csr_matrix(vect_combined))
    
        return vect
    
    def get_vectors_from_sentence(self, words, window,i):
        start_time=time.time()
        subwords=words[i:i+window]
        index= i + int(window/2)
        target= subwords[index:index+1]
        context= subwords[:index]+subwords[index+1:]
        vect_target=self.tokenize(target)
        vect_context= self.tokenize(context)
        vect_combined= np.hstack([vect_context,vect_target])
        return vect_combined





# Load le dataset

dataset = load_dataset("openwebtext", cache_dir="./")
dataset = dataset["train"]

# Load le vectorizer sauvegarder
path = "./save_massina_projet/vectors"
vect_saved=joblib.load("./save_massina_projet/fruitFlyVectorizer_window=10.pkl")
max_words=vect_saved.max_words
window=vect_saved.window
file_name = str(window)
vectorizer=FruitFlyVectorizer(max_words=max_words, window=window)
vectorizer.vect_dictionnary=vect_saved.vect_dictionnary
vectorizer.freq_dictionnary=vect_saved.freq_dictionnary

# Récupérer les vecteurs
print("="*100)
print("Transform")
vectorizer.parallel_transform(dataset, path=path, file_name=file_name)
print("="*100)

