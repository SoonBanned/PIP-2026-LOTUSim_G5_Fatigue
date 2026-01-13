import json
import os
from datetime import datetime


class Database:
    def __init__(self, chemin_fichier="users_db.json"):
        self.chemin_fichier = chemin_fichier
        self.data = {}
        # Si le fichier existe, on charge, sinon on crée un dictionnaire vide
        if os.path.exists(self.chemin_fichier):
            try:
                with open(self.chemin_fichier, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except json.JSONDecodeError:
                self.data = {}
        else:
            self.sauvegarder_db()

    def sauvegarder_db(self):
        """Méthode interne pour écrire dans le JSON"""
        with open(self.chemin_fichier, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)

    def enregistrerResultat(
        self, nom, scoreFatigueMentale, scoreFatiguePhysique, dateEtHeure, tempsReponse
    ):
        """
        Enregistre une session de test pour un utilisateur donné.
        Si l'utilisateur n'existe pas, il est créé.
        """
        nom = str(nom).strip()

        # Création de l'objet session
        nouvelle_session = {
            "fatigue_mentale": scoreFatigueMentale,
            "fatigue_physique": scoreFatiguePhysique,
            "date": dateEtHeure,  # Format string attendu ex: "2023-10-27 14:00"
            "temps_reponse_moyen": tempsReponse,
        }

        if nom not in self.data:
            self.data[nom] = []

        self.data[nom].append(nouvelle_session)
        self.sauvegarder_db()

    def enregistrerSession(self, nom: str, session: dict) -> None:
        """Enregistre une session complète de test (quiz + réponses + grading).

        Schéma recommandé pour `session`:
        {
          "test_id": str,
          "date": str (ISO),
          "totalTestTime": int,
          "quiz": list,
          "answers": list,
          "grading": dict,
        }
        """
        nom = str(nom).strip()
        if not nom:
            raise ValueError("Nom utilisateur vide")
        if not isinstance(session, dict):
            raise ValueError("session doit être un dict")

        # Ensure date exists
        if not session.get("date"):
            session = dict(session)
            session["date"] = datetime.now().isoformat(timespec="seconds")

        if nom not in self.data:
            self.data[nom] = []

        self.data[nom].append(session)
        self.sauvegarder_db()

    def recupererDerniereSession(self, nom: str) -> dict | None:
        nom = str(nom).strip()
        sessions = self.data.get(nom, [])
        if not sessions:
            return None
        if isinstance(sessions, list):
            return sessions[-1] if sessions else None
        return None

    def recupererSessionParTestId(self, nom: str, test_id: str) -> dict | None:
        nom = str(nom).strip()
        test_id = str(test_id).strip()
        sessions = self.data.get(nom, [])
        if not isinstance(sessions, list) or not test_id:
            return None
        for s in reversed(sessions):
            if isinstance(s, dict) and str(s.get("test_id") or "") == test_id:
                return s
        return None

    def recupererResultats(self, nom):
        """Retourne la liste des sessions d'un utilisateur ou une liste vide"""
        nom = str(nom).strip()
        return self.data.get(nom, [])

    def statistiqueAutre(self, nom):
        """
        Calcule la moyenne des scores des AUTRES utilisateurs (tout sauf 'nom').
        Retourne un dictionnaire avec les moyennes.
        """
        nom = str(nom).strip()
        total_mentale = 0
        total_physique = 0
        total_temps = 0
        count = 0

        for user_id, sessions in self.data.items():
            # On ignore l'utilisateur actuel pour comparer aux "autres"
            if user_id == nom:
                continue

            for session in sessions:
                total_mentale += session.get("fatigue_mentale", 0)
                total_physique += session.get("fatigue_physique", 0)
                total_temps += session.get("temps_reponse_moyen", 0)
                count += 1

        if count == 0:
            return {
                "moyenne_fatigue_mentale": 0,
                "moyenne_fatigue_physique": 0,
                "moyenne_temps_reponse": 0,
                "nombre_sessions_autres": 0,
            }

        return {
            "moyenne_fatigue_mentale": round(total_mentale / count, 2),
            "moyenne_fatigue_physique": round(total_physique / count, 2),
            "moyenne_temps_reponse": round(total_temps / count, 2),
            "nombre_sessions_autres": count,
        }
