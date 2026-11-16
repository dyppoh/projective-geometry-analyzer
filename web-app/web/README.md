# Finale – Weboberfläche

Starten, Ablauf und API: siehe [../README.md](../README.md).

```
src/
├── routes/
│   ├── +page.svelte              Start (Upload/Testbilder), Reiter „Linien & Gruppen“ / „Befunde“
│   ├── api/[...path]/+server.ts  reicht /api/* an den Python-Service durch (PYTHON_API_URL)
│   └── group/[...path]/+server.ts  öffentliche MLLM-Aufgaben /group/<token> -> Python-Service
└── lib/
    ├── Editor.svelte             automatisch annotieren, Linien auswählen/verschieben/zeichnen/löschen,
    │                             Gruppen anlegen/zuordnen, auswerten
    ├── Befunde.svelte            Urteil, Befund-Anteil, Befunde und Prüfgruppen auf dem Bild
    ├── MllmPanel.svelte          Adresse + Prompt für das MLLM, wartet auf die Abgabe
    ├── Canvas.svelte             Bild mit Zoom (Mausrad) und Verschieben (Ziehen)
    ├── server/proxy.ts           Durchreichen an den Python-Service
    ├── api.ts, types.ts, geometry.ts
    └── testbilder/               *.jpg/*.png - werden automatisch als Testbilder angeboten
```

Produktiv: `bun run build`, dann `bun run start` (liest `.env`, inkl. `BODY_SIZE_LIMIT=50M` für die base64-Bilder).
