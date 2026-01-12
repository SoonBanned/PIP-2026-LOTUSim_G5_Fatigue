from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from master import Master

app = FastAPI()
cerveau = Master()

# Sert les fichiers HTML/JS/CSS contenus dans le dossier "static"
# Assure-toi que web.html est dans un dossier nommé "static" ou modifie ici
app.mount("/interface", StaticFiles(directory="static", html=True), name="static")

class KeyEvent(BaseModel):
    key: str
    code: str
    type: str         # "up" ou "down"
    timestamp: int    # ms

class ReponseComplete(BaseModel):
    question_text: str
    reponse_text: str
    keystrokes_history: List[KeyEvent]

@app.get("/api/nouvelle_question")
def get_question():
    # Appel à la génération IA (peut prendre 2-5 secondes selon GPU)
    q = cerveau.generer_question_llm()
    return {"question": q}

@app.post("/api/analyser")
def post_reponse(data: ReponseComplete):
    print(f"Reçu {len(data.keystrokes_history)} événements clavier.")
    
    resultat = cerveau.analyser_reponse_et_fatigue(
        data.question_text, 
        data.reponse_text, 
        data.keystrokes_history
    )
    return resultat