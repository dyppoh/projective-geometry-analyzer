"""geometry-imitator: prozedurale 3D-Linien -> ControlNet-Bilder -> gelabelte DeepLSD-Linien.

Stufen (siehe pipeline.py / README.md):
    scenes   -> scene.json, lines.png, depth.png
    generate -> generations/<gen>/image.png
    detect   -> generations/<gen>/sample.json (nur angenommene)
"""

__version__ = "0.1.0"
