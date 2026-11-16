"""

HTTP-Service für die Weboberfläche (nur Standardbibliothek)

    python -m auswertung [--port 8765]

GET  /health                 {"status": "ok"}
POST /annotate               {"image": "<data:-URL>", "detector": "mlsd" | "hough"}
                                                          -> Scene (Gruppenstruktur aus M-LSD bzw. Hough + TLC)
POST /evaluate               Scene {imageWidth, imageHeight, lines, richtungen, horizonte, senkrecht}
                                                          -> Scene + verdict + checks

Gruppierung durch ein MLLM (mllm.py) - /group/... ist für das MLLM, über den Webserver öffentlich erreichbar:
POST /mllm                   {"image": "<data:-URL>", "detector"}   -> {"token", "lines", "shown", "detector"}
GET  /mllm/<token>           Stand: offen | abgegeben (+ Scene)
GET  /group/<token>          Aufgabenseite (HTML), dazu /bild.jpg, /linien.jpg, /ausschnitt-<1-4>.jpg,
                             /ausschnitt.jpg?x=&y=&w=&h=
POST /group/<token>          Gruppierung (JSON) -> Rückmeldung

Fehler als {"message": "..."} mit Status 400 (ungültige Eingabe), 404 (unbekannt) bzw. 500.
Absolute URLs auf der Aufgabenseite: Header X-Public-Origin (setzt der Webserver), sonst Host.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

from . import auswerten, mllm
from .annotieren import annotate, load

log = logging.getLogger("auswertung")


def _scene(body: dict) -> dict:
    for key in ("imageWidth", "imageHeight", "lines", "richtungen"):
        if key not in body:
            raise ValueError(f"'{key}' fehlt.")
    return body


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _handle(self, route) -> None:
        url = urlparse(self.path)
        parts = [unquote(p) for p in url.path.strip("/").split("/") if p]
        query = {k: v[-1] for k, v in parse_qs(url.query).items()}
        t0 = time.perf_counter()
        try:
            result = route(parts, query)
        except KeyError as e:                               # fehlendes Feld im JSON
            self._send(400, json.dumps({"message": f"{e.args[0]!r} fehlt."}).encode(), "application/json")
            return
        except LookupError as e:
            self._send(404, json.dumps({"message": str(e.args[0] if e.args else e)}).encode(), "application/json")
            return
        except (ValueError, TypeError) as e:
            self._send(400, json.dumps({"message": str(e)}, ensure_ascii=False).encode(), "application/json")
            return
        except Exception as e:  # noqa: BLE001 - Fehler an den Client melden statt Verbindung abzubrechen
            traceback.print_exc()
            self._send(500, json.dumps({"message": f"{type(e).__name__}: {e}"}).encode(), "application/json")
            return
        if isinstance(result, tuple):                       # (bytes | str, Content-Type)
            body, ct = result
            self._send(200, body.encode("utf-8") if isinstance(body, str) else body, ct)
        else:
            log.info("%s %s in %.0f ms", self.command, url.path, (time.perf_counter() - t0) * 1000)
            self._send(200, json.dumps(result, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _body(self) -> dict:
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}"
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:              # z.B. curl aus einer Windows-Konsole
            text = raw.decode("cp1252", errors="replace")
        body = json.loads(text)
        if not isinstance(body, dict):
            raise ValueError("JSON-Objekt erwartet.")
        return body

    def do_GET(self):
        def route(parts, query):
            if parts == ["health"]:
                return {"status": "ok"}
            if len(parts) == 2 and parts[0] == "mllm":
                return mllm.status(parts[1])
            if len(parts) == 2 and parts[0] == "group":
                origin = self.headers.get("X-Public-Origin") or f"http://{self.headers.get('Host')}"
                return mllm.page(parts[1], origin.rstrip("/")), "text/html; charset=utf-8"
            if len(parts) == 3 and parts[0] == "group":
                return mllm.image(parts[1], parts[2], query), "image/jpeg"
            raise LookupError(f"Unbekannter Pfad {self.path}")

        self._handle(route)

    def do_POST(self):
        def route(parts, query):
            if parts == ["annotate"]:
                body = self._body()
                return annotate(body["image"], body.get("detector") or "mlsd")
            if parts == ["evaluate"]:
                return auswerten(_scene(self._body()))
            if parts == ["mllm"]:
                body = self._body()
                return mllm.create(body["image"], body.get("detector") or "mlsd")
            if len(parts) == 2 and parts[0] == "group":
                return mllm.submit(parts[1], self._body())
            raise LookupError(f"Unbekannter Pfad {self.path}")

        self._handle(route)


class Server(ThreadingHTTPServer):
    # Unter Windows erlaubt SO_REUSEADDR, einen belegten Port ein zweites Mal zu binden - die Anfragen gehen dann
    # weiter an den alten Prozess. Lieber mit Fehlermeldung abbrechen.
    allow_reuse_address = sys.platform != "win32"


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m auswertung")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        httpd = Server((a.host, a.port), Handler)
    except OSError:
        sys.exit(f"Port {a.port} ist belegt?")
    log.info("Lade Modelle (M-LSD, TLC) ...")
    load()
    log.info("Auswertung läuft auf http://%s:%d", a.host, a.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


main()
