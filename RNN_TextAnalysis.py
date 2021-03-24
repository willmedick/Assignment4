"""
Authors: Jacob Flood and Will Medick
Name: RNN_TextAnalysis
Date: 3/22/21
"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import re
import nltk
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.callbacks import EarlyStopping
from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

#Read in csv file into pandas data frame
#Extract complaints and products into a new dataframe
df = pd.read_csv ('consumer_complaints.csv', usecols=['product','consumer_complaint_narrative'])

#Check how many values are null and count them
nullValues = df['consumer_complaint_narrative'].isna().sum()
print("Number of null values: " + str(nullValues))

#Check how many product values we have
productValues = df['consumer_complaint_narrative'].notna().sum()
print("Number of product values: " + str(productValues))

#Drop all empty values, reset the index, and drop them
drop = df.dropna(subset=['consumer_complaint_narrative'])
newdf0 = drop.reset_index(drop=True, inplace=False)
print(newdf0)
#Recount product values
productValues = df['consumer_complaint_narrative'].notna().sum()
print("Number of product values: " + str(productValues))
#Create a clean text function
def cleanText(text):
    
    #Lower case the text
    text = text.lower()
    #Compile pattern to remove all other characters
    pattern = re.compile(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-_]")
    #Sub the regular expresion with a "" character.
    text = re.sub(pattern, "", text)
    #Remove x from the text characters with a "" character.
    text = re.sub('x', "", text)
    #Split the text
    text = re.split(r'\W+', text)
    #For each word check if its a word and its an alphanumeric
    pattern2 = re.compile('[a-zA-Z0-9_]',re.ASCII)
    text2 = []
    for t in text:
        if (re.search(pattern2, t)):
            text2.append(t)    
    #Remove all english stop words
    stop_words = set(stopwords.words("english"))

    #Check if each word in the text and add the ones not in stop words
    newText = []
    for t in text2:
        if t not in stop_words:
            newText.append(t)
    #Join all the text by " "
    newText = " ".join(newText)
    #Return the clean text
    return newText

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
               filters = '[,.\"!@#$%^&*(){}?/;`~:<>+=-_]',
               lower = True,
               split = ' ',
               char_level = False)


#Fit Tokenizer object on the text
tf.fit_on_texts(newdf0["consumer_complaint_narrative"])

#Get the word index from tokenizer object
word_index = tf.word_index
#Print number of unique tokens found
print("Number of unique tokens: ", len(tf.word_counts))
#Get a text to sequences representation of the complaints
sequences= tf.texts_to_sequences(newdf0["consumer_complaint_narrative"])
#sequences= tf.text_to_word_sequence(newdf0["consumer_complaint_narrative"])

#Pad the sequences with the max length
data = pad_sequences(sequences, maxlen = maxCom, padding = 'post')
#Print the shape of the data
print(data.shape)
#Print the first example of the tokenizer object to the sequences to text
tokens = tf.sequences_to_texts(data)
print("First Example: " , tokens[0])
#print("First example: ", data[0])

#Get dummy representation of complaints into their one hot encoded vector categories
dummyList = []
for item in tokens:
    dummyList.append(tensorflow.keras.preprocessing.text.one_hot(item, maxWords))

#Split data into training and testing data with a test size of 10%
X_train, X_test, y_train, y_test = train_test_split(dummyList, test_size=0.10)
#Create your RNN model
model = Sequential()
#Add your embedding layers
model.add(Embedding(50,000, 250))
#Add your spatial dropout (20%)
model.add(SpatialDropout1D(.20))
#Add your LSTM layer
model.add(LSTM(100, dropout=.20, recurrent_dropout=.20))
#Add your dense layer
model.add(Dense(11, activation='softmax'))
#Compile your model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=[tf.keras.metrics.Accuracy()])
#Define epochs
epochs = 5
#Define batch size
batchSize = 64
#Fit the model with early stop training
model.fit(X_train, y_train, batch_size=batchSize, epochs=epochs, callbacks=EarlyStopping(monitor='loss', patience=3), min_delta=0.0001)
#Evaluate the model for accuracy
accuracy = tf.keras.metrics.Accuracy.result()
#Print the accuracy
print(accuracy)
#Show graph of loss over time for training data

#Show graph of accuracy over time for training data

