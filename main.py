from fastapi import FastAPI
from joblib import load as load_j
from nltk.stem import WordNetLemmatizer
app = FastAPI()
def lem(query,lemmatizer):
    return lemmatizer.lemmatize(query)
@app.get("/")
def root():
    return "Hello World!"

@app.get("/Chat-Bot")
def predict(qu:str):
    lemmatizer = WordNetLemmatizer()
    qu= lem(qu,lemmatizer)
    Pipe2=load_j('withoutTreatment_withSymp3.pkl')
    return str(Pipe2.predict([qu])[0])

