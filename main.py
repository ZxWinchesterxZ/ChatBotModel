from fastapi import FastAPI
from joblib import load as load_j
from nltk.stem import WordNetLemmatizer
import uvicorn
app = FastAPI()
port = os.environ["PORT"]
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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)


