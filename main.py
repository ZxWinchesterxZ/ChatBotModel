from fastapi import FastAPI
app = FastAPI()
from joblib import load as load_j
from nltk.stem import WordNetLemmatizer
def lem(query,lemmatizer):
    return lemmatizer.lemmatize(query)
@app.get("/")
async def root():
    return "Hello World!"

@app.get("/Chat-Bot")
async def predict(qu:str):
    lemmatizer = WordNetLemmatizer()
    qu= lem(qu,lemmatizer)
    Pipe2=load_j('withoutTreatment_withSymp3.pkl')
    return str(Pipe2.predict([qu])[0])


