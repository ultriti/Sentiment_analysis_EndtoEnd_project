from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path
from pydantic import BaseModel, Field

import re
import pickle
from keras.models import load_model

from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

"""
1. we are going to make the contrint like
A. Model Path (BiGRU)
B. Tokenizer Path
C. Max Sequnce Length = 50
D. Emotion Labels 
E. Emotion emoji
"""


# BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = "Artifacts"
STATIC_DIR = "static"

# Model Path Load
model_Path = str(r"Artifacts\BiGRU_Model.keras")

# Tokenizer Path Load
tokenizer_path = str(r"Artifacts\tokenizer.pkl")

# Mx Seq Len
max_seq_len = 50

# Emotion Labels
emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

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
        json_schema_extra={"example": "i am so much happy for this"},
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

dl_model = {}  # { 1. BiGRU model 2. Tokenizer } -> true if both loaded , {} -> false


@asynccontextmanager
async def Lifespan(app: FastAPI):
    print("loading the model and tokenizer...")

    dl_model["BiGRU"] = load_model(model_Path)  # BiGRU model

    with open(tokenizer_path, "rb") as file:
        dl_model["Tokenizer"] = pickle.load(file)  # tokenizer model

    print("models are loaded successfully UL..")

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
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# frontend mount here
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ------------------------
# api end points
@app.get("/", include_in_schema=False)
def serer_ui():
    return FileResponse(str(STATIC_DIR / "index.html"))


# hceck health of app route
@app.get("/health", response_model=HealthRepsonse)
def health_check():
    return HealthRepsonse(status="server is running UL!", model_loaded=bool(dl_model))


@app.post("/predict", response_model=PredictionResponse)
def prediction(text_input: TextInput):

    BiGRU_model = dl_model.get("BiGRU")
    tokenizer = dl_model.get("Tokenizer")

    if BiGRU_model is None or tokenizer is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded yet. Please load the model."
        )

    # filter the input

    text = preprocess_text(text_input.text)
    print(text)

    # tokenize the text
    tokenized_text = tokenizer.texts_to_sequences([text])

    # add padding
    padded_text = pad_sequences(
        tokenized_text, maxlen=max_seq_len, padding="post", truncating="post"
    )
    
    
    # sample_sequence = tokenizer.texts_to_sequences(sample_texts)
    # sample_padded_sequences = pad_sequences(sample_sequence,maxlen=50,padding='post',truncating='post')

    # sample_pred = np.argmax(model.predict(sample_padded_sequences),axis=1)

    # label_name = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    

    print(f"--------------------------padded_text:-\{padded_text}")   
    probabilities = BiGRU_model.predict(padded_text)[0]
    
    print(f"--------------------------probabilities:-\{probabilities}")
        
    predicted_index = int(np.argmax(probabilities))
    print(f"--------------------------predicted_index:-\{predicted_index}")
    
    all_probabilities = {
        label : float(prob) for label,prob in zip(emotion_labels,probabilities)
    }
    print(f"--------------------------predicted_label :-\{all_probabilities}")
    
    predicted_label = emotion_labels[predicted_index]
    # dict(zip(emotion_labels,probabilities))
    # xip join 2 object with thier indexes



    print(f"--------------------------predicted_label :-\{predicted_label}")
    
    return PredictionResponse(
        text=text_input.text,
        prediction_emotion=predicted_label,
        confidence=float(probabilities[predicted_index]),
        all_probabilities=all_probabilities
    )
