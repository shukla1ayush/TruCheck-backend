from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
from collections import Counter

app = FastAPI()

# CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this based on your frontend's actual URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

class TextRequest(BaseModel):
    text: str

def analyze_text(text: str):
    words = re.findall(r'\w+', text.lower())
    sentiment = sia.polarity_scores(text)
    
    score = min(
        0.4 * sum(1 for phrase in ["amazing", "best ever", "worst"] if phrase in text.lower()) / 2 +
        0.3 * len({word: count for word, count in Counter(words).items() if count > 3}) / 2 +
        0.3 * abs(sentiment['compound']),
        1.0
    )
    
    return {
        'is_fake': score > 0.65,
        'score': round(score, 2),
        'generic_phrases': sum(1 for phrase in ["amazing", "best ever", "worst"] if phrase in text.lower()),
        'repetition': {word: count for word, count in Counter(words).items() if count > 3}
    }

@app.post('/api/analyze')
async def analyze(request: TextRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail='Text cannot be empty')
    
    try:
        analysis = analyze_text(text)
        if analysis['is_fake']:
            analysis['explanation'] = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"Explain why this might be fake: {text}"}],
                max_tokens=50
            ).choices[0].message['content']
        
        return analysis
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import os
    openai.api_key = os.getenv('OPENAI_API_KEY')
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
