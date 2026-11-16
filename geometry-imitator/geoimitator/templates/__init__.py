from . import facade, objects, room, street

TEMPLATES = {
    "room": room.generate,
    "facade": facade.generate,
    "street": street.generate,
    "objects": objects.generate,
}

__all__ = ["TEMPLATES"]
