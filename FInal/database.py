import json
import os
import uuid
from datetime import datetime


class Database:
    def __init__(self, chemin_fichier="users_db.json"):
        self.chemin_fichier = chemin_fichier
        self.data = self._empty_db()

        # Si le fichier existe, on charge, sinon on crée une DB vide.
        if os.path.exists(self.chemin_fichier):
            try:
                with open(self.chemin_fichier, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict) and loaded.get("_schema_version") == 2:
                    self.data = loaded
                else:
                    # Old schema or invalid; start fresh.
                    self.data = self._empty_db()
                    # Overwrite the legacy file to keep storage light.
                    self.sauvegarder_db()
            except Exception:
                self.data = self._empty_db()
                # Overwrite corrupted/legacy file.
                self.sauvegarder_db()
        else:
            self.sauvegarder_db()

    @staticmethod
    def _empty_db() -> dict:
        return {"_schema_version": 2, "users": {}}

    def sauvegarder_db(self):
        """Méthode interne pour écrire dans le JSON"""
        with open(self.chemin_fichier, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)

    @staticmethod
    def _sanitize_name(nom: str) -> str:
        return " ".join(str(nom or "").strip().split())

    def _ensure_user(self, nom: str) -> dict:
        nom = self._sanitize_name(nom)
        if not nom:
            raise ValueError("Nom utilisateur vide")
        users = self.data.get("users")
        if not isinstance(users, dict):
            users = {}
            self.data["users"] = users
        user_obj = users.get(nom)
        if not isinstance(user_obj, dict):
            user_obj = {"sessions": []}
            users[nom] = user_obj
        if not isinstance(user_obj.get("sessions"), list):
            user_obj["sessions"] = []
        return user_obj

    def ajouter_session(
        self,
        nom: str,
        *,
        date_time: str,
        mental_fatigue: float | None,
        physical_fatigue: float | None,
        payload: dict,
        session_id: str | None = None,
    ) -> str:
        """Ajoute une session complète pour un utilisateur.

        Payload attendu: dict contenant au minimum quiz/answers/grading (et optionnellement meta/video/fatigue/totalTestTime).
        """
        nom = self._sanitize_name(nom)
        user_obj = self._ensure_user(nom)
        sessions = user_obj.get("sessions")
        if not isinstance(payload, dict):
            raise ValueError("payload doit être un dict")

        sid = str(session_id or uuid.uuid4().hex)
        dt = str(date_time or "").strip() or datetime.now().isoformat(
            timespec="seconds"
        )

        session = {
            "session_id": sid,
            "date_time": dt,
            "mental_fatigue": mental_fatigue,
            "physical_fatigue": physical_fatigue,
            "payload": payload,
        }
        sessions.append(session)
        self.sauvegarder_db()
        return sid

    def lister_sessions(self, nom: str) -> list[dict]:
        nom = self._sanitize_name(nom)
        if not nom:
            return []
        users = self.data.get("users")
        if not isinstance(users, dict):
            return []
        user_obj = users.get(nom)
        if not isinstance(user_obj, dict):
            return []
        sessions = user_obj.get("sessions")
        if not isinstance(sessions, list):
            return []
        return [s for s in sessions if isinstance(s, dict)]

    def recuperer_session(self, nom: str, session_id: str) -> dict | None:
        nom = self._sanitize_name(nom)
        session_id = str(session_id or "").strip()
        if not nom or not session_id:
            return None
        for s in reversed(self.lister_sessions(nom)):
            if str(s.get("session_id") or "") == session_id:
                return s
        return None

    def serie_autres_utilisateurs(self, nom: str) -> list[dict]:
        """Retourne une série journalière moyenne (mental/physical) des autres utilisateurs."""
        nom = self._sanitize_name(nom)
        users = self.data.get("users")
        if not isinstance(users, dict):
            return []

        buckets: dict[str, dict[str, float]] = {}
        counts: dict[str, int] = {}

        for user_name, user_obj in users.items():
            if user_name == nom or not isinstance(user_obj, dict):
                continue
            sessions = user_obj.get("sessions")
            if not isinstance(sessions, list):
                continue
            for s in sessions:
                if not isinstance(s, dict):
                    continue
                dt = str(s.get("date_time") or "")
                day = dt.split("T", 1)[0] if dt else ""
                if not day:
                    continue
                mf = s.get("mental_fatigue")
                pf = s.get("physical_fatigue")
                try:
                    mfv = float(mf) if mf is not None else None
                except Exception:
                    mfv = None
                try:
                    pfv = float(pf) if pf is not None else None
                except Exception:
                    pfv = None

                if day not in buckets:
                    buckets[day] = {"mental": 0.0, "physical": 0.0}
                    counts[day] = 0
                if mfv is not None:
                    buckets[day]["mental"] += mfv
                if pfv is not None:
                    buckets[day]["physical"] += pfv
                counts[day] += 1

        out: list[dict] = []
        for day in sorted(counts.keys()):
            c = counts[day] or 1
            out.append(
                {
                    "date": day,
                    "mental": round(buckets[day]["mental"] / c, 2),
                    "physical": round(buckets[day]["physical"] / c, 2),
                    "n": counts[day],
                }
            )
        return out

    # Back-compat helpers (not used anymore by the web UI)
    def recupererResultats(self, nom: str):
        return self.lister_sessions(nom)

    def statistiqueAutre(self, nom: str):
        # Simple overall average across other users.
        series = self.serie_autres_utilisateurs(nom)
        if not series:
            return {
                "moyenne_fatigue_mentale": 0,
                "moyenne_fatigue_physique": 0,
                "nombre_sessions_autres": 0,
            }
        total_n = sum(int(x.get("n") or 0) for x in series) or 0
        if total_n <= 0:
            return {
                "moyenne_fatigue_mentale": 0,
                "moyenne_fatigue_physique": 0,
                "nombre_sessions_autres": 0,
            }
        mental_sum = sum(
            float(x.get("mental") or 0) * int(x.get("n") or 0) for x in series
        )
        physical_sum = sum(
            float(x.get("physical") or 0) * int(x.get("n") or 0) for x in series
        )
        return {
            "moyenne_fatigue_mentale": round(mental_sum / total_n, 2),
            "moyenne_fatigue_physique": round(physical_sum / total_n, 2),
            "nombre_sessions_autres": total_n,
        }
