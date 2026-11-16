# Perspektivische Konsistenzprüfung


## Starten

```powershell
# Python-Service 
cd python
pip install -r requirements.txt
python -m auswertung          # http://127.0.0.1:8787

# Webserver
cd web
bun install
copy .env.example .env
bun run dev                 # http://localhost:5173
```

## Mit Docker

```sh
docker compose up -d --build    # http://localhost:3000
```
