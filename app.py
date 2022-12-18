#import discord
#import os
#from neuralintents import GenericAssistant
# from dotenv import load_dotenv

import uvicorn
from fastapi import FastAPI
import joblib
print(joblib.__version__)
import os
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import pickle
from pydantic import BaseModel

# chatbot=GenericAssistant('intents.json')
# chatbot.train_model()
# chatbot.save_model()
# load_dotenv()

model=open("emotion1.pkl","rb")
model_pred=joblib.load(model)

calc_vec=open("vector.pkl","rb")
vector=joblib.load(calc_vec)
#cv=CountVectorizer()


app=FastAPI()


@app.get('/')
async def index():
    return {"text":"Hai hemmooos"}


@app.get('/items/{text}')
async def getItems(text):
    return{"text":text}


@app.get('/predict/{text}')
async def predict(text):
    st=[]
    st.append(text)
    myvect=vector.transform(st).toarray()
    prediction=model_pred.predict(myvect)
    return {"text":text,"prediction":prediction[0]}





if __name__=='__main__':
    uvicorn.run(app,host="127.0.0.1",port=8525)
# client=discord.Client(intents=discord.Intents.default())
# @client.event
# async def on_message(message):
#     if message.author==client.user:
#         return
    
#     if (message.content.startswith("$aibot") & (message.content[7:]=="bye")):

#         response="https://youtu.be/mkDxuRvKUL8"
#         await message.channel.send(response)

#     if message.content.startswith("$aibot"):
#         response=chatbot.request(message.content[7:])
#         await message.channel.send(response)









# client.run(os.getenv('TOKEN'))