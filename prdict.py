import nltk
import numpy as np
import tensorflow
import pickle
import json
import random
from data_preprocessing import get_stem_words

ignore_words = ['?', '!',',','.', "'s", "'m"]

model = tensorflow.keras.models.load_model("chatbot_model.h5")
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

def ProcessingUserInput(input):
    wordToken1 = nltk.word_tokenize(input)
    wordToken2 = get_stem_words(wordToken1,ignore_words)
    wordToken2 = sorted(list(set(wordToken2)))
    bag = []
    bag_of_words = []
    for word in words:            
        if word in wordToken2:              
            bag_of_words.append(1)
        else:
            bag_of_words.append(0)
    bag.append(bag_of_words)
    return np.array(bag)

def botPrediction(user_input):
    input = ProcessingUserInput(user_input)
    prediction =  model.predict(input)
    predictionClass = np.argmax(prediction[0])
    return predictionClass
    
def botResponse(user_input):
    input = botPrediction(user_input)
    predicted_class = classes[input]
    for i in intents['intents']:
        if i["tag"] == predicted_class:  
            bot_response = random.choice(i['responses'])    
            return bot_response

print("Hi, I'm Chatbot, How may I help you?")  

while True :
    user_input=input("Please enter a message: ")
    print("User Input:",user_input)
    response = botResponse(user_input)
    print("bot response: ",response)