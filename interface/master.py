from llama_cpp import Llama
import json
import csv
import os
import datetime
import random
import re
import numpy as np  # Nécessaire pour les calculs statistiques (pip install numpy)
from pypdf import PdfReader

class Master:
    def __init__(self):
        print("--- Initialisation Master (LLAMA 3 - COLREG) ---")
        
        # 1. Chargement du Modèle
        self.llm = Llama(
            model_path="./modeles/Llama-3.2-3B-Instruct-Q6_K.gguf",
            #model_path="./modeles/meta-llama-3.1-8b.Q4_K_M.gguf",
            n_ctx=4096, 
            n_threads=4,
            n_gpu_layers=10, # Augmente si tu as une bonne VRAM
            verbose=False
        )

        # 2. Chargement PDF & Nettoyage
        self.colreg_text = ""
        try:
            reader = PdfReader("colreg.pdf")
            raw_text = ""
            # On lit un peu plus de pages pour avoir de la matière
            for page in reader.pages[:5]: 
                raw_text += page.extract_text()
            self.colreg_text = re.sub(r'\s+', ' ', raw_text).strip()
            print(f"--- PDF Chargé : {len(self.colreg_text)} caractères ---")
        except Exception as e:
            print(f"--- ERREUR PDF : {e} ---")
            self.colreg_text = "Règle 15 : Lorsque deux navires à propulsion mécanique font des routes qui se croisent..."

        # Stockage temporaire de la 'Vérité Terrain' pour la question en cours
        self.current_context_answer = ""
        
        self.csv_file = "donnees_vol.csv"
        self.init_csv()

    def init_csv(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, delimiter=';')
                writer.writerow(["Date", "Question", "Reponse_User", "Note", "Fatigue_Score", "Temps_Reponse"])

    def clean_json_output(self, raw_text):
        """Nettoie et RÉPARE la sortie du LLM pour extraire le JSON valide"""
        text = raw_text.strip()
        
        # 1. Nettoyage basique (retirer le markdown ```json s'il existe)
        text = text.replace("```json", "").replace("```", "")
        
        # 2. Tentative directe
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass # On continue si ça plante

        # 3. Tentative de réparation (Cas classique : manque "}")
        # On cherche la dernière accolade fermante
        last_brace = text.rfind('}')
        
        if last_brace == -1:
            # Aucune accolade fermante trouvée, on en ajoute une
            text += "}"
        
        try:
            return json.loads(text)
        except:
            # 4. Tentative désespérée (Cas : manque guillemet ET accolade)
            try:
                return json.loads(text + '" }')
            except:
                print(f"--- ECHEC PARSING JSON : {text} ---")
                return None

    def generer_question_llm(self):
        print(">>> Génération de question via LLM...")
        
        max_start = max(0, len(self.colreg_text) - 2000)
        start_idx = random.randint(0, max_start)
        excerpt = self.colreg_text[start_idx : start_idx + 2000]

        prompt = f"""<|start_header_id|>system<|end_header_id|>
Tu es un instructeur maritime. Utilise le texte ci-dessous pour créer une question.

Extrait:
"{excerpt}..."

Réponds avec ce format JSON strict :
{{
    "question": "La question...",
    "reponse_attendue": "L'explication..."
}}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{{"""
        try:
            # On demande un peu plus de tokens pour être sûr qu'il finisse
            output = self.llm(prompt, max_tokens=300, temperature=0.7, stop=["<|eot_id|>"])
            text_out = output['choices'][0]['text']
            
            # Reconstruction : On remet l'accolade ouvrante
            raw_json = "{" + text_out.strip()
            
            print(f"DEBUG RAW OUTPUT: {raw_json}") 

            data = self.clean_json_output(raw_json)
            
            if data and "question" in data:
                self.current_context_answer = data.get("reponse_attendue", "")
                return data["question"]
            else:
                # Fallback si vraiment illisible
                return "Erreur lecture. Question de secours : Quelle est la règle en cas de visibilité réduite ?"
                
        except Exception as e:
            print(f"Erreur Gen Question: {e}")
            return "Erreur système lors de la génération."

    def calculer_metriques_clavier(self, keystrokes):
        """
        Analyse les frappes pour détecter la fatigue.
        Métriques : 
        1. Flight Time (Temps entre le relâchement d'une touche et l'enfoncement de la suivante).
        2. Variance du Flight Time (Indicateur fort de fatigue cognitive).
        3. Nombre de corrections (Backspaces).
        """
        if not keystrokes or len(keystrokes) < 2:
            return 0, 0, 0

        flight_times = []
        backspaces = 0
        
        # Trier par timestamp pour être sûr
        sorted_keys = sorted(keystrokes, key=lambda x: x.timestamp)
        
        last_up_time = 0
        
        for k in sorted_keys:
            if k.key == "Backspace" and k.type == "down":
                backspaces += 1

            if k.type == "up":
                last_up_time = k.timestamp
            elif k.type == "down" and last_up_time > 0:
                flight = k.timestamp - last_up_time
                if flight < 2000: # On ignore les pauses trop longues (réflexion)
                    flight_times.append(flight)

        if not flight_times:
            return 0, 0, backspaces

        # Calculs statistiques
        avg_flight = np.mean(flight_times)
        variance_flight = np.var(flight_times) # C'est notre indicateur clé
        
        print(f"Stats Clavier -> Moyenne: {avg_flight:.1f}ms | Variance: {variance_flight:.1f} | Backspaces: {backspaces}")
        
        return avg_flight, variance_flight, backspaces

    def calculer_score_fatigue(self, note_ia, variance_frappe, backspaces):
        """
        Algorithme heuristique de fatigue (0 = Frais, 100 = Épuisé).
        Hypothèses :
        - Basse note = Fatigue cognitive.
        - Haute variance de frappe = Fatigue psychomotrice (irrégularité).
        - Beaucoup de corrections = Perte de concentration.
        """
        score = 0
        
        # 1. Impact de la note (Si note < 5/10, la fatigue augmente)
        if note_ia < 5:
            score += (10 - note_ia) * 5  # Max 50 pts
        
        # 2. Impact de la régularité (Variance > 5000ms² est suspect)
        # Normalisation arbitraire pour l'exemple
        if variance_frappe > 10000:
            score += 40
        elif variance_frappe > 5000:
            score += 20
            
        # 3. Impact des fautes de frappe
        score += (backspaces * 2) # 2 points de fatigue par correction

        return min(100, score) # Plafond à 100

    def analyser_reponse_et_fatigue(self, question, reponse_text, keystrokes_raw):
        print(">>> Début Analyse IA & Fatigue...")

        # A. Analyse Clavier
        avg_time, variance, nb_backspaces = self.calculer_metriques_clavier(keystrokes_raw)

        # B. Analyse Sémantique (LLM) avec AMORÇAGE JSON
        prompt_str = f"""<|start_header_id|>system<|end_header_id|>
Tu es un examinateur COLREG.
Texte référence : {self.colreg_text[:1000]}...
Réponse attendue : {self.current_context_answer}

Réponds UNIQUEMENT en JSON : {{"note": (int 0-10), "commentaire": (court)}}
<|eot_id|><|start_header_id|>user<|end_header_id|>
Question : "{question}"
Réponse élève : "{reponse_text}"
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{{"""

        try:
            output = self.llm(prompt_str, max_tokens=200, temperature=0.1)
            # Reconstruction du JSON
            raw_response = "{" + output['choices'][0]['text']
            resultat = self.clean_json_output(raw_response) or {"note": 0, "commentaire": "Erreur parsing IA"}
        except Exception as e:
            resultat = {"note": 0, "commentaire": f"Erreur IA: {e}"}

        note = resultat.get('note', 0)
        
        # C. Calcul Fatigue Global
        fatigue_score = self.calculer_score_fatigue(note, variance, nb_backspaces)
        
        # ... (Le reste de la fonction reste identique : calcul état fatigue et sauvegarde) ...
        
        etat_fatigue_str = "Frais"
        if fatigue_score > 30: etat_fatigue_str = "Légère fatigue"
        if fatigue_score > 60: etat_fatigue_str = "Fatigue Critique"

        with open(self.csv_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                question, reponse_text, note, fatigue_score, len(keystrokes_raw)
            ])

        return {
            "correction": note,
            "commentaire": resultat.get('commentaire'),
            "etat_fatigue": f"{etat_fatigue_str} (Score: {fatigue_score}/100)",
            "details_techniques": {
                "variance": round(variance, 2),
                "backspaces": nb_backspaces
            }
        }