from fastapi import FastAPI
import re

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}


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
    re.sub(r"'","",text)
    print(text)
    pass
    return
