from fruitFlyVectorizer import FruitFlyVectorizer, clean
from datasets import load_dataset
from tqdm import tqdm
import json
import csv
import joblib


# Load le dataset

dataset = load_dataset("openwebtext", cache_dir="./")
dataset = dataset["train"]

# Fit le vectorizer 

window = 10
max_words=2000
file_name = str(window)
path="./save_projet"
vectorizer=FruitFlyVectorizer(max_words=max_words, window=window)
print("="*100)
print("Fit")
vectorizer.parallel_fit(dataset,path, file_name=file_name)
print("="*100)
joblib.dump(vectorizer, path+"/fruitFlyVectorizer_window="+str(window)+".pkl")






