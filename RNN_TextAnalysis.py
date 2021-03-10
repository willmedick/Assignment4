import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import re
import nltk
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

#Read in csv file into pandas data frame
df = pd.read_csv ('consumer_complaints.csv', usecols=['product','sub_product','issue','sub_issue'])
#Extract complaints and products into a new dataframe

#Check how many values are null and count them
nullValues = df['sub_product'].isna().sum()
print("Number of null values: " + str(nullValues))

#Check how many product values we have
productValues = df['sub_product'].notna().sum()
print("Number of product values: " + str(productValues))

#Drop all empty values, reset the index, and drop them
drop = df.dropna(subset=['sub_product'])
newdf0 = drop.reset_index(drop=True, inplace=False)
#newdf = newdf0.drop(columns=['index'])
print(newdf0)
#Recount product values
productValues = df['sub_product'].notna().sum()
print("Number of product values: " + str(productValues))

#Create a clean text function

    #Lower case the text
    
    #Compile pattern to remove all other characters
    
    #Sub the regular expresion with a "" character.
    
    #Remove x from the text characters with a "" character.
    
    #Split the text
    
    #For each word check if its a word and its an alphanumeric
    
    #Remove all english stop words
    
    #Check if each word in the text and add the ones not in stop words
    
    #Join all the text by " "
    
    #Return the clean text
    

#Apply clean text to the complaints


#Define maximum number of words in our vocabulary to 50000

#Define maximum number of words within each complaint document to 250

#Define maximum number of words within each embedding to 100

#Implement Tokenizer object with num_words, filters, lower, split, and char_level

#Fit Tokenizer object on the text

#Get the word index from tokenizer object

#Print number of unique tokens found

#Get a text to sequences representation of the complaints

#Pad the sequences with the max length

#Print the shape of the data

#Print the first example of the tokenizer object to the sequences to text

    
