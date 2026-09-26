from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import re
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from keras.models import load_model
import pickle

"""
1. we are going to make the contrint like
A. Model Path (BiGRU)
B. Tokenizer Path
C. Max Sequnce Length = 50
D. Emotion Labels 
E. Emotion emoji
"""


# Model Path Load
model_Path = r"Artifacts\BiGRU_Model.keras"

# Tokenizer Path Load
tokenizer_path = r"Artifacts\tokenizer.pkl"

# Mx Seq Len
max_seq_len = 50

# Emotion Labels
emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# emotion emoji
emotion_emoji = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😯",
}

# ------------------
"""
2. Preprocess the text
A. Clean Raw text so it cna match the form used whil training
A - 1 . Convert text to lower case
B. Remove apostrophes (can't, don't) 
c. Remove special Characters / Punct
D. Remove extra space
"""


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"'", "", text)  # replace ' with empty str
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # remove punct
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces

    return text


# --------------------
"""
3. Request and Response Schemas
A. Text Input - String - input sent by user
B. Presiction Response 
C. Health Response ( Working / health checkup )
"""


class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="the sentence to analyze",
        json_schema_extra={"example": "i am happy fro this is working ! "},
    )


class PredictionResponse(BaseModel):
    text: str
    prediction_emotion: str
    confidence: float
    all_probabilities: dict[str, float]


class HealthRepsonse(BaseModel):
    status: str
    model_loaded: bool


"""
4. Model Loading and Life Span Management
Load the model and tokenizer once the server starts up.

"""

dl_model = {}


@asynccontextmanager
async def Lifespan(app: FastAPI):
    print("loading the model and tokenizer...")

    dl_model["BiGRU"] = load_model(model_Path)  # BiGRU model

    with open(tokenizer_path, "rb") as file:
        dl_model["Tokenizer"] = pickle.load(file)  # tokenizer model

    print("models are loaded successfully..")

    # to puase or stop the code or flow we use yeild

    yield  # model is paused but server is running and this point mdoel wait for request

    dl_model.clear()  # so when server i s closed /stopped (ctrl + c ) then remove artifacts from memory


"""
5. Mount the Static files to FastApi app
A. Enable cors origin shared 
B. 

"""

# main app ----------

app = FastAPI(lifespan=Lifespan)

app.add_middleware(
    CORSMiddleware, allow_origin=["*"], allow_methods=["*"], allow_headers=["*"]
)

app.mount('/frontend',StaticFiles(directory="frontend"),name='frontend')


# api end points
@app.get("/")
def read_root():
    return {"message": "Hello World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
