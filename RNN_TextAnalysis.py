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
df = pd.read_csv ('consumer_complaints.csv', usecols=['product','consumer_complaint_narrative'])
#Extract complaints and products into a new dataframe

#Check how many values are null and count them
nullValues = df['consumer_complaint_narrative'].isna().sum()
print("Number of null values: " + str(nullValues))

#Check how many product values we have
productValues = df['consumer_complaint_narrative'].notna().sum()
print("Number of product values: " + str(productValues))

#Drop all empty values, reset the index, and drop them
drop = df.dropna(subset=['consumer_complaint_narrative'])
newdf0 = drop.reset_index(drop=True, inplace=False)
#newdf = newdf0.drop(columns=['index'])
print(newdf0)
#Recount product values
productValues = df['consumer_complaint_narrative'].notna().sum()
print("Number of product values: " + str(productValues))
#Create a clean text function
def cleanText(text):
    
    #Lower case the text
    text = text.lower()
    #Compile pattern to remove all other characters
    pattern = re.compile(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]")
    #Sub the regular expresion with a "" character.
    text = re.sub(pattern, "", text)
    #Remove x from the text characters with a "" character.
    text = re.sub('x', "", text)
    #Split the text
    text = re.split(r'\W+', text)
    #For each word check if its a word and its an alphanumeric
    #text = re.sub('\w', "", text)
    
    #Remove all english stop words
    stop_words = set(stopwords.words("english"))
    #text = re.sub(stop_words, "", text)
    #text = re.sub(stopwords, "", text)
    
    #Check if each word in the text and add the ones not in stop words
    #Join all the text by " "
    
    #Return the clean text
    return text

#Apply clean text to the complaints
newdf0["consumer_complaint_narrative"] = newdf0["consumer_complaint_narrative"].apply(cleanText)



#Define maximum number of words in our vocabulary to 50000
maxWords = 50000
#Define maximum number of words within each complaint document to 250
maxCom = 250
#Define maximum number of words within each embedding to 100
maxEmbed = 100
#Implement Tokenizer object with num_words, filters, lower, split, and char_level
tf = Tokenizer(num_words = maxWords,
               filters = '[,.\"!@#$%^&*(){}?/;`~:<>+=-]',
               lower = True,
               split = ' ',
               char_level = True)

#Fit Tokenizer object on the text
newdf0["consumer_complaint_narrative"] = newdf0["consumer_complaint_narrative"].apply(tf.fit_on_texts)

#Get the word index from tokenizer object
words = tf.word_index
#Print number of unique tokens found
print("Number of unique tokens: ", len(set(words)))
#Get a text to sequences representation of the complaints
sequences = tf.texts_to_sequences(newdf0)
#Pad the sequences with the max length
data = pad_sequences(sequences, maxlen = maxCom, padding = 'post')
#Print the shape of the data
print(data.shape)
#Print the first example of the tokenizer object to the sequences to text

def main():
    print(newdf0)

if __name__ == "__main__":
    main()
