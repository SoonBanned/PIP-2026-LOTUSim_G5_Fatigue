from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from starlette.requests import Request


class MasterAPI(Protocol):
    async def on_results(self, payload: Dict[str, Any]) -> Dict[str, Any]: ...

    async def start_test(
        self, *, n: int = 4, difficulty: str = "easy"
    ) -> Dict[str, Any]: ...

    async def get_question(self, *, index: int) -> Dict[str, Any]: ...


class Interface:
    """Web interface runner (FastAPI + Uvicorn) designed to call back into Master.

    - Serves the HTML UI (Jinja2 template)
    - Exposes /save_results endpoint used by the front-end
    - Provides hooks for Master -> Interface calls (stop, set_questions)
    """

    def __init__(
        self,
        *,
        master: MasterAPI,
        host: str = "127.0.0.1",
        port: int = 5001,
        templates_dir: str | Path | None = None,
        assets_dir: str | Path | None = None,
    ) -> None:
        self.master = master
        self.host = host
        self.port = int(port)

        root = Path(__file__).resolve().parent
        self.templates_dir = Path(templates_dir) if templates_dir else (root / "web")
        self.assets_dir = Path(assets_dir) if assets_dir else (root / "web" / "assets")

        self._server: Any | None = None
        self._app: Any | None = None
        self._questions: list[dict[str, Any]] = []

    def set_questions(self, questions: list[dict[str, Any]]) -> None:
        """Optional hook for Master to provide questions to the UI later."""
        self._questions = list(questions or [])

    def create_app(self) -> Any:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import HTMLResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
        from fastapi.templating import Jinja2Templates
        from jinja2 import pass_context

        app = FastAPI(title="Fatigue Test")

        # Keep dev friction low (the front-end uses an absolute fetch URL).
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                f"http://{self.host}:{self.port}",
                f"http://localhost:{self.port}",
                f"http://127.0.0.1:{self.port}",
            ],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        if self.assets_dir.exists():
            app.mount(
                "/static", StaticFiles(directory=str(self.assets_dir)), name="static"
            )

        templates = Jinja2Templates(directory=str(self.templates_dir))

        # Compatibility layer for HTML templates originally written for Flask:
        #   {{ url_for('static', filename='logo.png') }}
        # Starlette's StaticFiles expects `path=...`.
        @pass_context
        def _url_for(ctx: Any, name: str, **params: Any) -> str:
            request: Request | None = ctx.get("request")
            if request is None:
                raise RuntimeError("Template context missing 'request'")

            if name == "static":
                filename = params.pop("filename", None)
                if filename is not None and "path" not in params:
                    params["path"] = filename
            return str(request.url_for(name, **params))

        templates.env.globals["url_for"] = _url_for

        @app.get("/", response_class=HTMLResponse)
        async def index(request: Request) -> Any:
            return templates.TemplateResponse(
                "Interface_V2.html",
                {"request": request},
            )

        @app.get("/health")
        async def health() -> Dict[str, Any]:
            return {"ok": True}

        @app.get("/api/questions")
        async def get_questions() -> Dict[str, Any]:
            return {"questions": self._questions}

        @app.post("/api/start_test")
        async def start_test(request: Request) -> JSONResponse:
            payload = await request.json()
            n = int(payload.get("n", 4))
            difficulty = str(payload.get("difficulty", "easy"))
            result = await self.master.start_test(n=n, difficulty=difficulty)
            return JSONResponse(result)

        @app.get("/api/question/{index}")
        async def get_question(index: int) -> JSONResponse:
            result = await self.master.get_question(index=int(index))
            return JSONResponse(result)

        @app.post("/save_results")
        async def save_results(request: Request) -> JSONResponse:
            payload = await request.json()
            result = await self.master.on_results(payload)
            return JSONResponse({"ok": True, "result": result})

        self._app = app
        return app

    async def serve(self) -> None:
        """Run Uvicorn in the current asyncio loop."""
        import uvicorn

        app = self._app or self.create_app()
        config = uvicorn.Config(
            app=app,
            host=self.host,
            port=self.port,
            loop="asyncio",
            log_level="info",
        )
        server = uvicorn.Server(config)
        self._server = server
        await server.serve()

    async def stop(self) -> None:
        """Request server shutdown (best-effort)."""
        server = self._server
        if server is None:
            return
        server.should_exit = True
        await asyncio.sleep(0)
