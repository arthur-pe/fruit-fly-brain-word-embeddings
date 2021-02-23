from fruitFlyVectorizer import FruitFlyVectorizer, clean
from datasets import load_dataset
from tqdm import tqdm
import json
import csv
import joblib
# Load le dataset

dataset = load_dataset("openwebtext", cache_dir="./")
dataset = dataset["train"]


# Load le vectorizer sauvegarder
path = "./save_projet/vectors_subdata/"
vect_saved=joblib.load("./save_projet/fruitFlyVectorizer_window=10.pkl")
max_words=vect_saved.max_words
window=vect_saved.window
file_name = str(window)+ "_words=2000_"
vectorizer=FruitFlyVectorizer(max_words=max_words, window=window)
vectorizer.vect_dictionnary=vect_saved.vect_dictionnary
vectorizer.freq_dictionnary=vect_saved.freq_dictionnary

# Récupérer les vecteurs
print("="*100)
print("Transform")
vectorizer.parallel_transform(dataset, path=path, file_name=file_name)
print("="*100)

