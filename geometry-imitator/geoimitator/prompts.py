"""
Zufällige Prompts pro Szenentyp

Ein Prompt setzt sich aus Motiv, Zustand, Materialien/Details, Ort, Jahreszeit,
Wetter/Licht, Farbstimmung, Kamera und Stil zusammen - jeder Teil ist optional,
die Reihenfolge wird gemischt, damit die Generierungen möglichst verschieden
aussehen. Zusätzlich (Einstellungen unter `generation.prompt`):

* ``offTopic`` - Anteil der Prompts mit einem Motiv OHNE Geometrie (Katze,
  Wald, Suppe, ...). Der Generator muss dann gegen das ControlNet arbeiten;
  die Linien landen in fremden Inhalten oder verschwinden.
* ``nonPhoto`` - Anteil der Prompts mit Render-/Kunststil statt Foto
"""

from __future__ import annotations

import numpy as np

SUBJECTS = {
    "room": [
        "a cozy living room", "a modern minimalist living room", "a scandinavian bedroom", "a hotel room",
        "a modern kitchen with cabinets", "an empty office", "an open-plan office with desks", "an industrial loft",
        "a classroom", "a small library with bookshelves", "a bathroom with tiled walls", "a hospital corridor room",
        "a warehouse interior", "a dentist waiting room", "an art gallery room", "a student dorm room",
        "a japanese tatami room", "an old victorian parlor", "a basement storage room", "a children's playroom",
        "a conference room", "a retro 70s living room", "a concrete brutalist interior", "a museum hall",
        "a laundromat", "a gym with mirrors", "a small cafe interior", "a bakery shop interior", "a barber shop",
        "a hostel dormitory", "a train station waiting hall", "a church interior", "a recording studio",
        "a server room", "a garage workshop", "a hallway of an apartment", "an attic room", "a laboratory",
        "a restaurant dining room", "a bar with a counter", "a yoga studio", "a boutique clothing store",
        "a supermarket aisle", "a pharmacy", "a hotel lobby", "a government office", "a soviet-era kitchen",
        "a moroccan riad room", "a mid-century modern study", "a home office", "a newly renovated apartment",
        "an abandoned room", "an empty apartment before moving in", "a room under renovation",
        "a prison cell", "a sauna", "a swimming pool hall", "a squash court", "a spa treatment room",
    ],
    "facade": [
        "a modern apartment building", "an old european townhouse", "a red brick warehouse", "a glass office building",
        "a suburban family house", "a soviet-era apartment block", "a colorful row of houses in amsterdam",
        "a japanese residential building", "a historic building with stone facade", "a small shop with a storefront",
        "a school building", "a hotel facade", "a parisian haussmann building", "an american suburban street house",
        "a concrete parking garage", "a new york brownstone", "a mediterranean whitewashed house", "a factory building",
        "a bauhaus building", "a berlin altbau", "a viennese apartment building", "a georgian terrace house",
        "a scandinavian wooden house", "a half-timbered house", "a hospital building", "a university building",
        "a train station building", "a police station", "a post office", "a modern library facade",
        "a mixed-use building with shops", "a run-down tenement", "a renovated loft building", "a public housing block",
        "a colonial building in havana", "a building in lisbon with azulejo tiles", "an office tower base",
        "a hong kong residential tower", "a seoul apartment complex", "a singapore shophouse", "a dutch canal house",
        "a russian khrushchyovka", "a mexican house with painted walls", "a motel", "a gas station", "a fire station",
        "a city hall", "a museum facade", "a cinema facade", "a warehouse converted to offices", "a farmhouse",
    ],
    "street": [
        "a city street in paris", "a busy street in new york", "a quiet residential street", "a street in tokyo",
        "a main street in a small european town", "a street in berlin with apartment buildings", "a london street",
        "an empty street early in the morning", "a street in barcelona", "a downtown street with shops",
        "a street in an industrial area", "a street in a chinese city", "a rainy city street", "a street in amsterdam",
        "a street in rome", "a street in vienna", "a street in prague", "a street in copenhagen", "a street in madrid",
        "a street in istanbul", "a street in buenos aires", "a street in mexico city", "a street in montreal",
        "a street in seoul", "a street in osaka", "a street in hong kong", "a street in san francisco",
        "a street in chicago", "a street in a suburban neighborhood", "a street in a village",
        "a street in an old town with cobblestones", "a pedestrian zone", "a shopping street", "a back alley",
        "a street with a tram line", "a boulevard with trees", "a street in a business district",
        "a street in a soviet-era district", "a street in a new development area", "a street during a festival",
        "a street on a sunday morning", "a street with construction work", "a street at a market day",
    ],
    "objects": [
        "cardboard boxes on the floor", "wooden crates in a warehouse", "a stack of books on a table",
        "furniture in a showroom", "packages in a storage room", "concrete blocks on a construction site",
        "a wooden staircase and boxes", "minimalist product photo of geometric objects", "shipping boxes in a garage",
        "a workshop with wooden boxes and shelves", "toy blocks on a carpet", "gift boxes on a wooden floor",
        "storage shelves with bins", "moving boxes in an empty apartment", "pallets in a warehouse",
        "a desk with drawers", "metal lockers", "a stack of suitcases", "plastic crates with vegetables",
        "a wooden bench and a table", "a kitchen counter with appliances", "a bookshelf with books",
        "a cabinet and a chest of drawers", "boxes on a loading dock", "speakers and amplifiers on a stage",
        "a stack of bricks", "wooden planks and beams", "an electrical cabinet", "vending machines", "a pile of crates",
        "shoe boxes on shelves", "a sculpture made of cubes", "a modular sofa", "server racks",
    ],
}

# Motive ganz ohne Geometrie (für offTopic)
OFF_TOPIC = [
    "a close-up portrait of a cat", "a golden retriever lying in the grass", "a bowl of ramen", "a plate of pasta",
    "a dense forest with fog", "ocean waves at the beach", "a field of sunflowers", "a mountain landscape",
    "a portrait of an old man", "a portrait of a young woman with freckles", "a coral reef with fish",
    "clouds in the sky", "abstract colorful paint swirls", "a close-up of a flower", "a desert with sand dunes",
    "a bowl of fruit", "a jellyfish in dark water", "autumn leaves on the ground", "a horse in a meadow",
    "a waterfall in the jungle", "a close-up of tree bark", "a galaxy and stars", "a parrot on a branch",
    "a lake with reflections", "a birthday cake", "a bunch of grapes", "a frog on a leaf", "snow-covered pine trees",
    "a macro photo of an eye", "a cup of coffee with latte art",
]

CONDITION = {
    "room": ["brand new", "well kept", "lived-in", "messy", "slightly worn", "abandoned and dusty", "freshly painted",
             "under renovation", "luxurious", "cheap and simple", "vintage", "sterile and clean"],
    "facade": ["brand new", "well maintained", "weathered", "run-down", "freshly renovated", "with graffiti",
               "with peeling paint", "historic and restored", "abandoned", "partly covered with ivy", "dirty"],
    "street": ["clean", "busy", "deserted", "run-down", "well-kept", "touristy", "gentrified", "historic",
               "under construction", "with puddles", "full of litter"],
    "objects": ["new", "used", "old and dusty", "damaged", "neatly arranged", "messy", "stacked", "scattered"],
}

DETAILS = {
    "room": ["wooden floor", "tiled floor", "concrete floor", "white walls", "plants", "colorful walls", "carpet",
             "large windows", "minimalist decor", "cluttered with everyday objects", "parquet floor", "marble floor",
             "terrazzo floor", "linoleum floor", "exposed brick wall", "wood paneling", "wallpaper with patterns",
             "posters on the walls", "framed pictures", "ceiling lamps", "pendant lights", "curtains", "blinds",
             "a rug", "bookshelves", "potted plants", "a sofa", "chairs", "cables on the floor", "a radiator",
             "exposed pipes on the ceiling", "fluorescent ceiling lights", "wooden beams", "green walls",
             "pastel colors", "dark wood furniture", "white furniture", "metal furniture"],
    "facade": ["brick", "plaster", "concrete", "glass and steel", "stone", "painted wood", "balconies", "trees",
               "red brick", "yellow plaster", "white render", "sandstone", "granite", "timber cladding",
               "corrugated metal", "ceramic tiles", "flower boxes on the windows", "air conditioning units",
               "satellite dishes", "shutters", "awnings", "a shop sign", "graffiti", "ivy", "scaffolding",
               "street lamps", "parked bicycles", "a bus stop", "drainpipes", "laundry hanging from windows"],
    "street": ["parked cars", "trees along the sidewalk", "pedestrians far away", "shop signs", "street lamps",
               "bicycles", "traffic signs", "a bus", "a tram", "taxis", "delivery vans", "scooters",
               "outdoor cafe tables", "market stalls", "construction fences", "bollards", "trash bins", "benches",
               "overhead wires", "traffic lights", "billboards", "neon signs", "flags", "planters", "puddles",
               "fallen leaves", "manhole covers", "road markings", "a fire hydrant"],
    "objects": ["wooden texture", "cardboard texture", "clean background", "concrete floor", "studio lighting",
                "painted metal", "plastic", "plywood", "labels and tape", "dust", "scratches", "a white wall",
                "a tiled floor", "a wooden floor", "colorful paint", "black and white objects", "shadows on the floor"],
}

PLACES = ["in germany", "in france", "in italy", "in spain", "in portugal", "in the netherlands", "in poland",
          "in sweden", "in the uk", "in the usa", "in canada", "in japan", "in china", "in south korea", "in india",
          "in brazil", "in mexico", "in morocco", "in turkey", "in russia", "in australia", "in south africa",
          "in vietnam", "in egypt", "in argentina"]

SEASON = ["in spring", "in summer", "in autumn", "in winter", "with snow", "with blossoming trees",
          "with autumn leaves"]

LIGHT = {
    "indoor": ["soft daylight from the windows", "warm evening light", "bright overhead lights", "dim lamp light",
               "natural light", "overcast daylight", "sunlight with hard shadows", "candle light",
               "cold fluorescent light", "neon light", "night, lights on", "morning sun rays", "backlit by the window",
               "dramatic light and shadow", "soft diffuse light", "colored LED lights", "light through blinds",
               "golden afternoon light", "dark, only a single lamp", "flash photography"],
    "outdoor": ["sunny day", "overcast sky", "golden hour", "blue hour", "after rain, wet ground", "winter, light snow",
                "midday sun with hard shadows", "foggy morning", "night with street lights", "heavy rain",
                "thunderstorm clouds", "sunset", "sunrise", "hazy summer day", "clear blue sky", "low sun, long shadows",
                "night, neon lights", "dusk", "misty", "snowstorm", "bright noon", "cloudy with sun breaks",
                "moonlight", "smog"],
}

MOOD = ["warm colors", "cool colors", "muted colors", "vibrant colors", "pastel tones", "earthy tones",
        "high contrast", "low contrast", "desaturated", "teal and orange", "monochrome", "black and white",
        "sepia tones", "cinematic color grading", "bright and airy", "dark and moody"]

CAMERA = ["wide angle", "ultra wide angle", "35mm", "50mm", "24mm", "eye level", "low angle", "slightly from above",
          "shallow depth of field", "deep focus", "long exposure", "handheld", "tilted horizon", "symmetrical view",
          "street level view", "from across the street", "from a corner of the room"]

STYLES = {
    "photo": ["photograph", "DSLR photo", "architectural photography", "smartphone photo", "real estate photo",
              "interior design magazine photo", "documentary photo", "film photo, kodak portra", "wide angle photo",
              "high quality photo, sharp focus", "editorial photo", "polaroid photo", "disposable camera photo",
              "35mm film photo, grain", "fujifilm photo", "leica photo", "iphone photo", "travel photo",
              "street photography", "snapshot", "old photo from the 1980s", "old photo from the 1960s",
              "security camera footage", "drone photo", "hdr photo", "instagram photo", "stock photo"],
    "render": ["3d render", "octane render", "unreal engine", "architectural visualization", "blender render",
               "video game screenshot", "isometric render", "clay render", "low poly render"],
    "art": ["oil painting", "watercolor painting", "pencil sketch", "ink drawing", "digital painting",
            "concept art", "anime background", "comic style", "impressionist painting", "gouache illustration",
            "vintage postcard", "linocut print"],
}

QUALITY = ["realistic, highly detailed", "highly detailed", "realistic", "8k", "sharp", "masterpiece",
           "professional", "award winning", "detailed textures", "natural colors"]

NEGATIVE_CORE = ["warped lines", "bent walls", "distorted"]
NEGATIVE_EXTRA = ["blurry", "lowres", "text", "watermark", "logo", "frame", "border", "oversaturated",
                  "people in foreground", "jpeg artifacts", "deformed", "noise", "out of focus", "cropped",
                  "ugly", "bad quality", "duplicate", "fisheye"]
NEGATIVE_NON_PHOTO = ["cartoon", "drawing", "painting", "illustration", "3d render", "cgi"]


def _pick(rng: np.random.Generator, items, p: float = 1.0) -> list[str]:
    return [str(rng.choice(items))] if rng.random() < p else []


def make_prompt(scene_type: str, rng: np.random.Generator, off_topic: float = 0.04,
                non_photo: float = 0.08) -> dict:
    """-> {prompt, negativePrompt, subject, offTopic, style}"""
    indoor = scene_type in ("room", "objects")
    off = rng.random() < off_topic
    r = rng.random()
    style = "photo" if r >= non_photo else ("render" if r < non_photo / 2 else "art")

    if off:
        subject = str(rng.choice(OFF_TOPIC))
        parts = _pick(rng, MOOD, 0.4) + _pick(rng, CAMERA, 0.3)
    else:
        subject = str(rng.choice(SUBJECTS.get(scene_type, SUBJECTS["objects"])))
        details = DETAILS.get(scene_type, DETAILS["objects"])
        parts = list(rng.choice(details, size=int(rng.integers(0, 4)), replace=False))
        parts += _pick(rng, CONDITION.get(scene_type, CONDITION["objects"]), 0.4)
        parts += _pick(rng, LIGHT["indoor" if indoor else "outdoor"], 0.85)
        parts += _pick(rng, MOOD, 0.35) + _pick(rng, CAMERA, 0.35)
        if not indoor:
            parts += _pick(rng, PLACES, 0.25) + _pick(rng, SEASON, 0.3)
    parts = [str(p) for p in parts]
    rng.shuffle(parts)
    parts = [subject] + parts + _pick(rng, STYLES[style], 0.9) + _pick(rng, QUALITY, 0.6)

    if rng.random() < 0.1:
        negative = ""
    else:
        neg = [n for n in NEGATIVE_CORE if rng.random() < 0.8]
        neg += [str(n) for n in rng.choice(NEGATIVE_EXTRA, size=int(rng.integers(2, 9)), replace=False)]
        if style == "photo":
            neg += NEGATIVE_NON_PHOTO
        negative = ", ".join(neg)
    return {"prompt": ", ".join(parts), "negativePrompt": negative, "subject": subject, "offTopic": bool(off),
            "style": style}






# Negativ: Motive ohne gerade Kanten / Bauwerke

NEG_CATEGORIES: dict[str, dict[str, list[str]]] = {
    "landscape": {
        "subjects": [
            "a mountain landscape", "a dense forest", "a misty forest", "rolling green hills", "a desert with sand dunes",
            "a rocky canyon", "a volcano", "a glacier", "a meadow with wildflowers", "a misty swamp", "a rainforest",
            "a bamboo forest", "a field of lavender", "a wheat field", "a field of sunflowers", "a tundra",
            "a savanna with acacia trees", "a moorland with heather", "a birch forest", "a pine forest",
            "an autumn forest", "a snowy mountain peak", "a valley with a winding river", "a cave with stalactites",
            "a mossy rock formation", "a salt flat", "red rock formations", "a jungle clearing", "a steppe",
            "an alpine meadow", "a bog", "a forest floor with ferns", "a hill covered in fog", "a mangrove forest",
            "lava fields", "a sand desert at dusk", "an oasis with palm trees", "a field of poppies",
            "a riverbank with willows", "a bush landscape", "a highland with sheep",
        ],
        "details": ["untouched", "wild", "vast", "remote", "overgrown", "lush", "barren", "rugged", "peaceful",
                    "dramatic", "gentle"],
        "settings": ["in the distance", "under a wide sky", "with low clouds", "with a rainbow", "with grazing animals",
                     "with patches of snow", "covered in fog", "after a storm"],
        "camera": ["wide angle", "telephoto compression", "aerial view", "drone shot", "panorama", "eye level",
                   "low angle"],
    },
    "water": {
        "subjects": [
            "ocean waves", "a stormy sea", "a calm lake", "a mountain lake with reflections", "a waterfall",
            "a river in a canyon", "a coral reef", "a tropical lagoon", "rain on a pond", "a jellyfish in dark water",
            "a school of fish", "a kelp forest", "a sea turtle underwater", "surf breaking on rocks",
            "a frozen lake with cracks in the ice", "a mountain stream", "rapids", "a tide pool", "a whirlpool",
            "a hot spring with steam", "the surface of water with ripples", "underwater sun rays", "a dolphin jumping",
            "a whale underwater", "bubbles underwater", "sea foam on the sand", "a wave curling", "a misty river",
            "a lily pond", "an iceberg", "a geyser", "a reflection of clouds in a puddle",
        ],
        "details": ["turquoise", "crystal clear", "murky", "glittering", "foamy", "deep blue", "emerald green",
                    "dark", "still", "turbulent"],
        "settings": ["at sunrise", "under a cloudy sky", "seen from above", "seen from below the surface",
                     "with mist", "surrounded by rocks", "at night under the moon"],
        "camera": ["long exposure", "underwater photo", "macro", "telephoto", "aerial view", "slow shutter",
                   "fast shutter, frozen motion"],
    },
    "sky": {
        "subjects": [
            "clouds in the sky", "a thunderstorm", "lightning in the night sky", "a sunset", "a sunrise",
            "the northern lights", "a starry night sky", "the milky way", "a galaxy", "a nebula", "a full moon",
            "a solar eclipse", "storm clouds", "cumulus clouds", "a rainbow", "a tornado", "fog rolling over hills",
            "a pink sky at dusk", "mammatus clouds", "a supercell storm", "a comet", "the surface of the moon",
            "a planet with rings", "a sandstorm", "falling snow", "heavy rain", "sun rays through clouds",
            "noctilucent clouds", "a halo around the sun", "drifting smoke in the sky",
        ],
        "details": ["dramatic", "glowing", "colorful", "ominous", "soft", "swirling", "vivid", "faint", "majestic"],
        "settings": ["over the sea", "over a forest", "over the desert", "over the mountains", "over an open field"],
        "camera": ["wide angle", "astrophotography", "long exposure", "time lapse still", "telephoto",
                   "looking straight up"],
    },
    "plants": {
        "subjects": [
            "a close-up of a flower", "a rose", "a sunflower", "a bouquet of tulips", "a cherry blossom branch",
            "a lotus flower", "an orchid", "a dandelion seed head", "ferns", "a mushroom on the forest floor",
            "moss on a stone", "lichen on bark", "a cactus", "succulents", "an apple tree with apples", "autumn leaves",
            "a single leaf with dew drops", "ivy leaves", "a weeping willow", "an old oak tree", "a palm leaf",
            "a field of tulips", "wild berries", "a pine cone", "a sprouting seedling", "seaweed",
            "a poppy flower", "a hydrangea", "lavender flowers", "a water lily", "a thistle", "a clover meadow",
            "a magnolia blossom", "grass blades with frost", "tree roots", "a bonsai tree", "a carnivorous plant",
            "a fruit hanging from a branch", "fallen petals", "a vine with grapes",
        ],
        "details": ["delicate", "blooming", "wilted", "fresh", "wet", "frosted", "vibrant", "pale", "dried",
                    "backlit", "fuzzy"],
        "settings": ["in a garden", "in the wild", "in a meadow", "against a blurred background", "on dark soil",
                     "in the rain", "in morning dew", "in a forest"],
        "camera": ["macro", "extreme close-up", "shallow depth of field", "bokeh", "50mm", "top-down view"],
    },
    "animals": {
        "subjects": [
            "a cat", "a dog", "a golden retriever", "a horse", "a lion", "a tiger", "an elephant", "a giraffe",
            "a zebra", "a fox", "a wolf", "a bear", "a deer", "a rabbit", "a squirrel", "a hedgehog", "an owl",
            "an eagle", "a parrot", "a hummingbird", "a flamingo", "a penguin", "a peacock", "a swan", "a duck",
            "a frog", "a snake", "a chameleon", "a lizard", "a turtle", "a butterfly", "a bee on a flower",
            "a dragonfly", "a ladybug", "a snail", "a spider web with dew", "a koala", "a panda", "a monkey",
            "a sheep", "a cow in a meadow", "a highland cow", "a goat", "a seal", "an octopus", "a seahorse",
            "a red panda", "a sloth", "a kitten", "a puppy", "a flock of birds", "a herd of wild horses",
            "a hamster", "an alpaca", "a camel in the desert", "a moose", "a crab on the beach",
        ],
        "details": ["sleeping", "curious", "running", "playful", "wet", "fluffy", "young", "old", "resting",
                    "yawning", "hunting", "grooming itself", "looking at the camera"],
        "settings": ["in tall grass", "in the snow", "on a rock", "in the forest", "in the water", "on a branch",
                     "in a meadow", "in the jungle", "on the beach", "in the savanna", "in the mud"],
        "camera": ["wildlife photography", "telephoto lens", "close-up portrait", "shallow depth of field",
                   "eye level", "macro", "action shot"],
    },
    "people": {
        "subjects": [
            "a portrait of an old man", "a portrait of a young woman with freckles", "a child laughing",
            "a bearded man", "an elderly woman smiling", "a teenager with curly hair", "a baby sleeping",
            "a face with face paint", "a woman with long red hair", "a man with tattoos", "a dancer in motion",
            "a runner", "a swimmer underwater", "a surfer on a wave", "a hiker on a mountain ridge",
            "two friends hugging", "a mother holding a baby", "a farmer in a field", "a fisherman at the sea",
            "a girl blowing dandelion seeds", "a boy playing in the mud", "a musician singing", "hands in soil",
            "close-up of an eye", "close-up of lips", "a silhouette against the sunset", "a woman in a flower field",
            "a man in the rain", "a face half in shadow", "an athlete sweating",
        ],
        "details": ["smiling", "serious", "thoughtful", "joyful", "tired", "freckled", "wrinkled", "sunburned",
                    "wet hair", "windblown hair"],
        "settings": ["in nature", "in a field", "on the beach", "in the forest", "against a plain background",
                     "against the sky", "in the snow", "in a garden"],
        "camera": ["85mm portrait", "close-up", "shallow depth of field", "candid", "natural light portrait",
                   "low key portrait", "high key portrait"],
    },
    "food": {
        "subjects": [
            "a bowl of fruit", "a plate of pasta", "a bowl of ramen", "a pizza", "sushi", "a burger", "a croissant",
            "a stack of pancakes", "a bowl of salad", "a birthday cake", "chocolate truffles", "strawberries",
            "a bunch of grapes", "sliced oranges", "a pile of apples", "avocado halves", "a curry", "dumplings",
            "a bowl of soup", "fresh bread", "a cheese platter", "ice cream", "macarons", "cookies", "a fruit tart",
            "honey dripping", "melted chocolate", "a smoothie bowl", "spaghetti", "a tomato salad", "popcorn",
            "cherries", "a cinnamon roll", "fried eggs", "a paella", "tacos", "noodles in broth", "a donut",
            "spices in a heap", "coffee beans", "blueberries", "a watermelon slice", "a pomegranate",
        ],
        "details": ["freshly made", "steaming", "glossy", "messy", "colorful", "homemade", "rustic", "juicy",
                    "crispy", "dusted with sugar"],
        "settings": ["close-up", "on a dark background", "on a linen cloth", "with herbs", "with crumbs around",
                     "in soft light"],
        "camera": ["food photography", "macro", "overhead shot", "45 degree angle", "shallow depth of field"],
    },
    "texture": {
        "subjects": [
            "tree bark", "moss", "sand ripples", "pebbles", "wet sand with footprints", "a cracked dry riverbed",
            "frost patterns", "animal fur", "feathers", "snake skin", "fish scales", "rough stone", "lava rock",
            "dried leaves", "a pile of autumn leaves", "flower petals", "grass", "snow with tracks", "ice crystals",
            "foam", "mud", "rust", "lichen", "coral", "sea sponge", "wool", "a sheepskin", "crumpled fabric",
            "silk folds", "velvet", "soap bubbles", "water droplets", "salt crystals in a heap", "clay",
            "marble swirls", "agate slice", "pine needles", "seeds", "cotton", "a honey comb close-up",
        ],
        "details": ["close-up", "highly detailed", "natural", "weathered", "wet", "dry", "soft", "rough",
                    "organic", "irregular"],
        "settings": ["filling the frame", "in natural light", "with soft shadows", "slightly out of focus at the edges"],
        "camera": ["macro", "top-down view", "extreme close-up", "texture photo", "shallow depth of field"],
    },
    "abstract": {
        "subjects": [
            "abstract colorful paint swirls", "ink drops in water", "smoke swirls", "liquid marbling",
            "oil and water bubbles", "bokeh lights", "a fluid art painting", "colored powder explosion",
            "flowing silk in the air", "light painting curves", "a watercolor wash", "soap film colors",
            "a fractal of organic shapes", "melting colors", "a nebula-like cloud of paint", "glowing plasma",
            "aurora-like light waves", "splashing milk", "a burst of confetti", "wavy gradients", "alcohol ink art",
            "acrylic pour", "drifting fog with colored light", "a swirl of sand", "floating bubbles",
            "blurred colorful lights bokeh", "a kaleidoscope of flowers", "rippling fabric", "dripping paint",
            "colorful clouds of dye",
        ],
        "details": ["vibrant", "pastel", "neon", "monochrome", "iridescent", "dreamy", "chaotic", "soft",
                    "glowing", "psychedelic"],
        "settings": ["on a black background", "on a white background", "in motion", "frozen in time"],
        "camera": ["macro", "high speed photography", "long exposure", "studio shot"],
    },
}

NEG_COMMON = {
    "light": ["golden hour", "soft daylight", "overcast light", "dramatic light", "backlight", "rim light",
              "dappled light", "harsh sunlight", "moonlight", "candle light", "studio light", "blue hour",
              "warm light", "cold light", "low light", "side light"],
    "weather": ["after rain", "in light rain", "in fog", "in snowfall", "on a windy day", "on a hot day",
                "with mist", "under storm clouds"],
    "season": ["in spring", "in summer", "in autumn", "in winter"],
}
OUTDOOR_CATEGORIES = {"landscape", "water", "sky", "plants", "animals"}
# Stile ohne Architektur-/Straßen-Bezug
NEG_STYLES = {k: [s for s in v if not any(w in s for w in ("architectur", "real estate", "interior", "street",
                                                            "security camera", "drone", "isometric", "video game"))]
              for k, v in STYLES.items()}

NEGATIVE_GEOMETRY = [
    "buildings", "architecture", "house", "skyscraper", "city", "street", "road", "bridge", "walls", "windows",
    "doors", "furniture", "room", "interior", "table", "shelves", "boxes", "fence", "railing", "stairs", "poles",
    "power lines", "wires", "cars", "vehicles", "signs", "text", "letters", "grid", "tiles", "stripes",
    "checkerboard", "geometric pattern", "straight lines", "frame", "picture frame", "man-made objects",
]


def make_negative_prompt(rng: np.random.Generator, non_photo: float = 0.08) -> dict:
    """Prompt ohne Geometrie-Motiv; Bauwerke und gerade Kanten im Negativ-Prompt.
    -> {prompt, negativePrompt, subject, category, offTopic, style}"""
    r = rng.random()
    style = "photo" if r >= non_photo else ("render" if r < non_photo / 2 else "art")
    category = str(rng.choice(list(NEG_CATEGORIES)))
    c = NEG_CATEGORIES[category]
    subject = str(rng.choice(c["subjects"]))
    parts = _pick(rng, c["details"], 0.5) + _pick(rng, c["settings"], 0.6) + _pick(rng, NEG_COMMON["light"], 0.5) \
        + _pick(rng, MOOD, 0.35) + _pick(rng, c["camera"], 0.5)
    if category in OUTDOOR_CATEGORIES:
        parts += _pick(rng, NEG_COMMON["weather"], 0.25) + _pick(rng, NEG_COMMON["season"], 0.25)
    parts = [str(p) for p in parts]
    rng.shuffle(parts)
    parts = [subject] + parts + _pick(rng, NEG_STYLES[style], 0.9) + _pick(rng, QUALITY, 0.6)
    neg = [n for n in NEGATIVE_GEOMETRY if rng.random() < 0.4]
    neg += [str(n) for n in rng.choice(NEGATIVE_EXTRA, size=int(rng.integers(1, 6)), replace=False)]
    if style == "photo":
        neg += NEGATIVE_NON_PHOTO
    return {"prompt": ", ".join(parts), "negativePrompt": ", ".join(neg), "subject": subject, "category": category,
            "offTopic": True, "style": style}
