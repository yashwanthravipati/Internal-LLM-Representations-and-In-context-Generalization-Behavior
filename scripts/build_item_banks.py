"""T-011: build the curated item banks in data/items/ from the lists below.

Source for all lists: curated by the project (agent-drafted 2026-10-26, pending the student's
spot-check). Ground-truth rules are in configs/domains.yaml. Items whose membership is
debatable are marked confidence=medium/low with a note; main analyses use `high` only.

    uv run python scripts/build_item_banks.py
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "items"
SOURCE = "curated-v1"


def words(s: str) -> list[str]:
    return [w.strip() for w in s.split(",") if w.strip()]


# ---------------------------------------------------------------------------
# Experiment 1: categories (superordinate -> category -> items)
# Rules: lowercase singular common nouns; each item in exactly one category; no proper nouns,
# no abstract nouns. "medium:" prefix marks a debatable member.
# ---------------------------------------------------------------------------

CATEGORIES: dict[str, dict[str, str]] = {
    "animal": {
        "mammal": """dog, cat, horse, cow, pig, sheep, goat, lion, tiger, bear, wolf, fox, deer,
            rabbit, mouse, rat, elephant, giraffe, zebra, monkey, gorilla, kangaroo, camel,
            squirrel, hamster, donkey, leopard, cheetah, hippo, rhino, otter, beaver, raccoon,
            moose, bison, panda, koala, hedgehog, badger, chimpanzee, llama, skunk, mole, weasel,
            jaguar, antelope""",
        "bird": """eagle, sparrow, robin, crow, pigeon, owl, parrot, penguin, swan, duck, goose,
            chicken, turkey, hawk, falcon, pelican, flamingo, ostrich, peacock, seagull, raven,
            canary, finch, woodpecker, heron, stork, vulture, hummingbird, magpie, dove, crane,
            emu, quail, kingfisher, albatross, puffin, swallow, starling, blackbird, wren, jay,
            cardinal, toucan, cuckoo, lark""",
        "fish": """salmon, trout, tuna, cod, shark, goldfish, carp, catfish, herring, sardine,
            mackerel, eel, perch, pike, bass, halibut, swordfish, piranha, anchovy, haddock,
            tilapia, stingray, minnow, guppy, barracuda, sturgeon, flounder, marlin, snapper,
            grouper, clownfish, mullet, sole, plaice, bream, whiting, pollock, monkfish""",
        "insect": """ant, bee, wasp, fly, mosquito, butterfly, moth, beetle, ladybug, grasshopper,
            cricket, dragonfly, cockroach, termite, flea, louse, hornet, caterpillar, firefly,
            locust, cicada, mantis, weevil, gnat, bumblebee, earwig, aphid, midge, horsefly,
            silverfish, stick insect, damselfly""",
        "reptile": """snake, lizard, crocodile, alligator, turtle, tortoise, iguana, gecko,
            chameleon, cobra, python, viper, rattlesnake, komodo, anaconda, boa, skink, terrapin,
            gharial, monitor, adder, mamba, caiman, tuatara""",
    },
    "plant": {
        "fruit": """apple, banana, orange, pear, grape, cherry, peach, plum, mango, pineapple,
            strawberry, lemon, lime, watermelon, melon, kiwi, apricot, raspberry, blueberry, fig,
            coconut, papaya, pomegranate, grapefruit, blackberry, cranberry, nectarine, tangerine,
            date, lychee, guava, passionfruit, gooseberry, currant, quince, persimmon,
            cantaloupe, clementine""",
        "vegetable": """carrot, potato, onion, cabbage, lettuce, spinach, broccoli, cauliflower,
            celery, pea, bean, cucumber, garlic, radish, turnip, beet, leek, asparagus, kale,
            parsnip, zucchini, artichoke, sprout, eggplant, okra, yam, squash, corn, chard, shallot,
            rutabaga, fennel, arugula, watercress, endive, kohlrabi""",
        "tree": """oak, pine, maple, birch, willow, elm, cedar, spruce, fir, ash, beech, palm,
            redwood, sequoia, cypress, poplar, sycamore, chestnut, walnut, larch, yew, hemlock,
            magnolia, eucalyptus, baobab, juniper, alder, hawthorn, hazel, linden, aspen,
            mahogany, teak, acacia, hickory""",
        "flower": """rose, tulip, daisy, lily, sunflower, orchid, daffodil, violet, poppy,
            carnation, lavender, jasmine, iris, marigold, dandelion, peony, lilac, hyacinth,
            chrysanthemum, begonia, geranium, petunia, pansy, azalea, camellia, gardenia,
            buttercup, snowdrop, crocus, bluebell, foxglove, hibiscus, zinnia, dahlia, aster,
            primrose, freesia, gladiolus""",
        "herb": """basil, parsley, mint, rosemary, thyme, sage, oregano, dill, cilantro, chive,
            tarragon, coriander, marjoram, bay, lemongrass, chervil, sorrel, lovage, borage""",
    },
    "food": {
        "dessert": """cake, pie, cookie, brownie, pudding, cupcake, muffin, doughnut, tart,
            cheesecake, custard, mousse, sorbet, fudge, trifle, macaron, eclair, pastry,
            meringue, gelato, waffle, pancake, crumble, souffle, tiramisu, parfait, sundae,
            truffle, toffee, cobbler, strudel, baklava""",
        "drink": """water, milk, coffee, tea, juice, beer, wine, soda, lemonade, cocoa, whiskey,
            vodka, rum, gin, brandy, cider, champagne, smoothie, milkshake, espresso,
            cappuccino, latte, tequila, sake, kombucha, punch, liqueur, sherry, mead,
            cocktail""",
        "dish": """pizza, pasta, sandwich, burger, soup, salad, stew, curry, sushi, taco,
            burrito, lasagna, omelette, risotto, paella, casserole, noodle, dumpling, kebab,
            chowder, goulash, quiche, ravioli, spaghetti, chili, falafel, hotdog, enchilada,
            moussaka, ramen""",
    },
    "artifact": {
        "vehicle": """car, bus, truck, train, bicycle, motorcycle, airplane, helicopter, boat,
            ship, taxi, van, tractor, scooter, tram, subway, yacht, canoe, kayak, submarine,
            jeep, limousine, ambulance, bulldozer, sailboat, ferry, glider, rocket, wagon,
            sled, carriage, trolley, minivan, hovercraft, caravan""",
        "tool": """hammer, screwdriver, wrench, saw, drill, pliers, chisel, shovel, rake, axe,
            spade, crowbar, mallet, file, level, hoe, trowel, clamp, vise, sander, scraper,
            awl, ladder, pickaxe, sledgehammer, hacksaw, tape, stapler, pitchfork, wheelbarrow""",
        "furniture": """chair, table, sofa, bed, desk, shelf, cabinet, dresser, wardrobe, stool,
            bench, couch, bookcase, armchair, nightstand, cupboard, ottoman, recliner, futon,
            crib, hammock, sideboard, credenza, chaise, cot, footstool, bureau, armoire, vanity,
            divan""",
        "clothing": """shirt, pants, dress, skirt, jacket, coat, sweater, jeans, shorts, blouse,
            vest, suit, hoodie, cardigan, pajamas, robe, gown, tuxedo, overalls, leggings,
            sweatshirt, tunic, poncho, kimono, sarong, blazer, raincoat, trousers, jumpsuit,
            parka, tracksuit, apron""",
        "footwear": """shoe, boot, sandal, slipper, sneaker, loafer, clog, moccasin, flipflop,
            stiletto, pump, wader, espadrille, mule, oxford, brogue, galosh, trainer, heel,
            platform""",
        "kitchenware": """spoon, fork, knife, plate, bowl, cup, mug, pan, pot, kettle, ladle,
            spatula, whisk, colander, grater, peeler, skillet, wok, tongs, saucepan, teapot,
            jug, pitcher, sieve, rolling pin, cleaver, platter, tray, saucer, corkscrew""",
        "instrument": """piano, guitar, violin, drum, flute, trumpet, saxophone, cello, clarinet,
            harp, trombone, tuba, oboe, banjo, ukulele, accordion, harmonica, bagpipe, organ,
            xylophone, tambourine, bassoon, mandolin, harpsichord, cymbal, piccolo, sitar,
            lute, viola, bugle""",
        "weapon": """sword, gun, rifle, pistol, spear, dagger, bow, arrow, cannon, grenade,
            missile, bomb, shotgun, crossbow, musket, revolver, mace, halberd, lance, sabre,
            cutlass, javelin, catapult, slingshot, harpoon, bayonet, trident, rapier, machete,
            torpedo""",
        "toy": """doll, ball, kite, yoyo, puzzle, teddy, rattle, marble, top, balloon, frisbee,
            puppet, slinky, lego, robot, dollhouse, pinwheel, jigsaw, skateboard, trampoline,
            playdough, kaleidoscope, seesaw, swing, hula hoop""",
        "device": """phone, computer, laptop, tablet, television, radio, camera, printer,
            keyboard, mouse pad, microwave, refrigerator, toaster, blender, dishwasher, washer,
            dryer, router, speaker, headphone, microphone, projector, scanner, monitor,
            calculator, smartwatch, console, modem, thermostat, vacuum""",
        "building": """house, church, school, hospital, castle, tower, barn, cottage, palace,
            temple, mosque, cathedral, skyscraper, warehouse, factory, library, museum, prison,
            stadium, theater, hotel, apartment, cabin, mansion, hut, lighthouse, garage,
            chapel, bungalow, fortress, igloo, pagoda""",
        "container": """box, bag, basket, bottle, jar, can, bucket, barrel, crate, chest,
            suitcase, backpack, envelope, sack, carton, tin, vase, urn, purse, wallet, tub,
            canister, drum, trunk, pouch, flask, hamper, bin, briefcase, satchel""",
    },
    "body": {
        "body part": """head, arm, leg, hand, foot, finger, toe, eye, ear, nose, mouth, lip,
            tooth, tongue, neck, shoulder, elbow, knee, ankle, wrist, chin, cheek, forehead,
            hip, thumb, heel, chest, belly, back, thigh, eyebrow, eyelid, scalp, palm, calf,
            shin""",
    },
    "person": {
        "profession": """doctor, teacher, nurse, lawyer, engineer, farmer, chef, pilot, baker,
            plumber, carpenter, dentist, artist, writer, singer, actor, soldier, sailor,
            painter, butcher, tailor, firefighter, police officer, judge, scientist, banker,
            librarian, architect, mechanic, electrician, cashier, journalist, surgeon,
            pharmacist, accountant, gardener, janitor, waiter, barber""",
    },
    "nature": {
        "weather": """rain, snow, hail, sleet, fog, mist, wind, storm, thunder, lightning,
            drizzle, frost, sunshine, tornado, hurricane, blizzard, cyclone, monsoon, typhoon,
            breeze, gale, drought, heatwave, rainbow, dew, cloud""",
        "gem": """diamond, ruby, emerald, sapphire, pearl, opal, topaz, amethyst, garnet, jade,
            onyx, turquoise, quartz, amber, jasper, agate, peridot, aquamarine, tourmaline,
            moonstone, zircon, coral, obsidian, lapis""",
        "landform": """mountain, hill, valley, river, lake, ocean, island, desert, forest,
            canyon, cliff, volcano, glacier, plateau, beach, cave, waterfall, swamp, marsh,
            peninsula, bay, delta, dune, meadow, prairie, gorge, ridge, lagoon, fjord, crater""",
        "metal": """iron, gold, silver, copper, tin, lead, zinc, aluminum, nickel, platinum,
            titanium, mercury, chromium, cobalt, magnesium, tungsten, uranium, lithium,
            sodium, potassium, calcium, steel, bronze, brass, pewter""",
    },
}

# Debatable memberships: item -> (confidence, note). Applied wherever the item appears.
DEBATABLE = {
    "mouse": ("medium", "also a computer device"),
    "date": ("low", "polysemous (calendar date)"),
    "bass": ("medium", "also a musical instrument / voice"),
    "sole": ("medium", "also part of a shoe/foot"),
    "monitor": ("low", "monitor lizard; polysemous (screen)"),
    "file": ("medium", "polysemous (document)"),
    "level": ("low", "polysemous"),
    "tape": ("medium", "polysemous"),
    "top": ("low", "spinning top; polysemous"),
    "swing": ("medium", "playground swing; polysemous"),
    "drum": ("medium", "instrument and container"),
    "pump": ("low", "shoe sense is rare"),
    "mule": ("low", "animal and shoe"),
    "oxford": ("low", "proper noun sense dominates"),
    "platform": ("low", "shoe sense is rare"),
    "trainer": ("medium", "also a person"),
    "heel": ("low", "body part and shoe part"),
    "back": ("low", "polysemous"),
    "palm": ("medium", "tree and body part"),
    "calf": ("medium", "body part and young cow"),
    "bay": ("medium", "herb (bay leaf) and landform"),
    "tin": ("medium", "metal and container"),
    "lead": ("medium", "polysemous"),
    "crane": ("medium", "bird and machine"),
    "jay": ("medium", "short, polysemous"),
    "cardinal": ("medium", "also a clergy rank"),
    "seal": ("medium", "polysemous"),
    "punch": ("medium", "polysemous"),
    "chili": ("medium", "dish and pepper"),
    "corn": ("medium", "botanically a grain"),
    "squash": ("medium", "also a sport"),
    "pea": ("high", ""),
    "bean": ("high", ""),
    "sprout": ("medium", "short for brussels sprout"),
    "coral": ("low", "animal colony, not a gem"),
    "amber": ("medium", "fossil resin, not a mineral"),
    "pearl": ("medium", "organic gem"),
    "tray": ("medium", "serving item"),
    "vacuum": ("medium", "polysemous"),
    "console": ("medium", "polysemous"),
    "speaker": ("medium", "also a person"),
    "mercury": ("medium", "also a planet"),
    "steel": ("medium", "alloy"),
    "bronze": ("medium", "alloy"),
    "brass": ("medium", "alloy; also instruments"),
    "pewter": ("medium", "alloy"),
    "lego": ("medium", "brand name"),
    "slinky": ("low", "brand name"),
    "playdough": ("low", "brand-like"),
    "robot": ("medium", "not only a toy"),
    "cloud": ("medium", "object rather than phenomenon"),
    "rainbow": ("medium", "optical phenomenon"),
    "drought": ("medium", "condition rather than event"),
    "cobbler": ("medium", "dessert and shoemaker"),
    "truffle": ("medium", "chocolate and fungus"),
    "pastry": ("medium", "category rather than item"),
    "pancake": ("medium", "breakfast rather than dessert"),
    "waffle": ("medium", "breakfast rather than dessert"),
    "cocktail": ("medium", "category of drink"),
    "noodle": ("medium", "ingredient rather than dish"),
    "hotdog": ("medium", "usually two words"),
    "arrow": ("medium", "projectile"),
    "bow": ("medium", "polysemous"),
    "pitcher": ("medium", "also a baseball player"),
    "jug": ("high", ""),
    "chest": ("medium", "container and body part"),
    "trunk": ("medium", "container, tree part, elephant part"),
    "can": ("medium", "also a modal verb"),
    "bin": ("high", ""),
    "washer": ("medium", "polysemous (hardware)"),
    "dryer": ("high", ""),
    "organ": ("medium", "also a body organ"),
    "ash": ("medium", "polysemous"),
    "fir": ("high", ""),
    "yew": ("high", ""),
    "hazel": ("medium", "also a colour/name"),
    "iris": ("medium", "also part of the eye"),
    "violet": ("medium", "also a colour"),
    "rose": ("medium", "also a verb/colour"),
    "lily": ("medium", "also a name"),
    "sage": ("medium", "polysemous"),
    "mint": ("medium", "polysemous"),
    "orange": ("medium", "also a colour"),
    "lime": ("medium", "also a colour/mineral"),
    "turkey": ("medium", "also a country"),
    "chicken": ("high", ""),
    "duck": ("medium", "also a verb"),
    "fly": ("medium", "also a verb"),
    "kiwi": ("medium", "also a bird"),
    "bat": ("medium", "also sports equipment"),
    "python": ("medium", "also a programming language"),
    "adder": ("medium", "polysemous"),
    "boa": ("medium", "also a scarf"),
    "louse": ("high", ""),
    "monkfish": ("high", ""),
    "plaice": ("medium", "rare word"),
    "mole": ("medium", "polysemous"),
    "deer": ("high", ""),
}

# ---------------------------------------------------------------------------
# Experiment 2: crossed 2x2 domains
# ---------------------------------------------------------------------------

# Animals: A = class (mammal / bird); B = domestication (domestic / wild).
# Domestic = a species (or breed) kept and bred by humans as livestock or pets, whose typical
# referent is the domestic animal. Wild = no domesticated form.
ANIMALS = {
    ("mammal", "domestic"): ("""dog, cat, cow, horse, pig, sheep, goat, donkey, hamster, llama,
        alpaca, ferret, gerbil, mule, ox, poodle, puppy, kitten, pony, bull""", "high"),
    ("mammal", "wild"): ("""lion, tiger, wolf, bear, fox, deer, elephant, giraffe, zebra,
        monkey, gorilla, kangaroo, leopard, cheetah, hippo, rhino, otter, beaver, raccoon, moose,
        squirrel, badger, panda, koala, hedgehog, chimpanzee, skunk, jaguar, antelope, bison""",
        "high"),
    ("bird", "domestic"): ("""chicken, hen, rooster, turkey, canary, budgie, parakeet, cockatiel,
        goose, duck, quail, peacock, lovebird, pigeon, chick, duckling""", "high"),
    ("bird", "wild"): ("""eagle, sparrow, robin, crow, owl, penguin, hawk, falcon, pelican,
        flamingo, seagull, raven, woodpecker, heron, stork, vulture, hummingbird, magpie,
        albatross, puffin, kingfisher, toucan, wren, swallow, starling, finch, cuckoo, ostrich,
        emu, lark""", "high"),
}
# Overrides within the animals domain (confidence, note).
ANIMAL_NOTES = {
    "goose": ("medium", "domestic and wild geese both common"),
    "duck": ("medium", "domestic and wild (mallard) both common"),
    "pigeon": ("medium", "domestic/feral; also wild rock doves"),
    "turkey": ("medium", "wild turkeys exist; also a country"),
    "peacock": ("low", "kept ornamentally but largely wild"),
    "quail": ("medium", "farmed and wild"),
    "parakeet": ("medium", "pet budgie, but many wild parakeet species"),
    "lovebird": ("medium", "pet trade and wild"),
    "chick": ("medium", "young bird, usually chicken"),
    "duckling": ("medium", "young duck"),
    "llama": ("high", ""),
    "alpaca": ("high", ""),
    "ostrich": ("medium", "farmed, but no domestic form"),
    "emu": ("medium", "farmed, but no domestic form"),
    "bison": ("medium", "some ranched"),
    "bull": ("medium", "male cattle"),
    "puppy": ("medium", "young dog"),
    "kitten": ("medium", "young cat"),
    "poodle": ("medium", "breed rather than species"),
    "squirrel": ("high", ""),
    "finch": ("medium", "some kept as pets"),
    "canary": ("high", ""),
}

# Countries: A = continent (Europe / Asia); B = coastal / landlocked (no sea coastline;
# Caspian Sea coastline counts as landlocked, following standard lists).
# Transcontinental countries (Russia, Turkey, Kazakhstan, Georgia, Azerbaijan, Armenia, Cyprus,
# Egypt) are excluded except where noted.
COUNTRIES = {
    ("Europe", "coastal"): ("""France, Spain, Italy, Germany, Portugal, Greece, Norway, Sweden,
        Finland, Denmark, Ireland, Netherlands, Belgium, Poland, Croatia, Romania, Bulgaria,
        Albania, Estonia, Latvia, Lithuania, Iceland, Ukraine, Slovenia, Montenegro, Malta,
        Monaco, England, Scotland, Wales""", "high"),
    ("Europe", "landlocked"): ("""Austria, Switzerland, Hungary, Slovakia, Czechia, Serbia,
        Belarus, Moldova, Luxembourg, Liechtenstein, Andorra, Macedonia, Kosovo, San Marino,
        Vatican""", "high"),
    ("Asia", "coastal"): ("""China, Japan, India, Korea, Vietnam, Thailand, Indonesia,
        Malaysia, Philippines, Singapore, Cambodia, Myanmar, Bangladesh, Pakistan, Iran, Iraq,
        Israel, Lebanon, Syria, Jordan, Yemen, Oman, Qatar, Kuwait, Bahrain, Taiwan, Sri Lanka,
        Brunei, Saudi Arabia""", "high"),
    ("Asia", "landlocked"): ("""Mongolia, Nepal, Bhutan, Laos, Afghanistan, Kyrgyzstan,
        Tajikistan, Turkmenistan, Uzbekistan, Kazakhstan""", "high"),
}
COUNTRY_NOTES = {
    "England": ("medium", "constituent country of the UK"),
    "Scotland": ("medium", "constituent country of the UK"),
    "Wales": ("medium", "constituent country of the UK"),
    "Ukraine": ("high", ""),
    "Jordan": ("medium", "tiny coastline (Gulf of Aqaba); also a name"),
    "Korea": ("medium", "ambiguous between North and South (both coastal)"),
    "Kazakhstan": ("medium", "transcontinental; Caspian-only coastline"),
    "Turkmenistan": ("medium", "Caspian-only coastline"),
    "Macedonia": ("medium", "officially North Macedonia"),
    "Kosovo": ("medium", "partially recognized"),
    "Vatican": ("medium", "city-state"),
    "Iraq": ("medium", "very short coastline"),
    "Taiwan": ("medium", "political status"),
    "Bosnia": ("low", "20 km coastline"),
}

# Words: A = semantic category (animal / food); B = first letter (b / p).
WORDS = {
    ("animal", "b"): ("""bear, bee, bat, beaver, bison, bull, badger, baboon, boar, buffalo,
        butterfly, beetle, bunny, bobcat, budgie, bulldog, barracuda, buzzard, bluebird""",
        "high"),
    ("animal", "p"): ("""pig, panda, parrot, penguin, puma, pony, poodle, pigeon, python,
        porcupine, peacock, pelican, panther, puppy, possum, piranha, puffin, platypus, pheasant,
        prawn""", "high"),
    ("food", "b"): ("""bread, butter, banana, bean, bacon, biscuit, broccoli, burger, bagel,
        berry, beef, beet, brownie, bun, burrito, barley, basil, brioche, bratwurst""",
        "high"),
    ("food", "p"): ("""pie, pizza, pasta, pear, peach, plum, pea, potato, pepper, pancake,
        popcorn, pork, pretzel, pudding, pumpkin, peanut, pineapple, parsnip, pickle,
        pastry""", "high"),
}
WORD_NOTES = {"bat": ("medium", "polysemous"), "basil": ("medium", "herb"),
              "berry": ("medium", "category rather than item"),
              "prawn": ("medium", "crustacean; also food"),
              "pepper": ("medium", "vegetable and spice"),
              "buzzard": ("medium", "rarer word")}


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def build_categories() -> int:
    rows, seen = [], {}
    for sup, cats in CATEGORIES.items():
        for cat, items in cats.items():
            for item in words(items):
                if item in seen:
                    # Keep the first assignment; record the clash so it is visible.
                    rows[seen[item]][4] = "low"
                    rows[seen[item]][5] += f"; also listed under {cat}"
                    continue
                conf, note = DEBATABLE.get(item, ("high", ""))
                seen[item] = len(rows)
                rows.append([item, cat, sup, SOURCE, conf, note])
    write_csv(OUT / "categories.csv",
              ["item", "category", "superordinate", "source", "confidence", "notes"], rows)
    return len(rows)


def build_crossed(name: str, spec: dict, notes: dict, attr_a: str, attr_b: str) -> int:
    rows, seen = [], set()
    for (va, vb), (items, default_conf) in spec.items():
        for item in words(items):
            if item in seen:
                raise ValueError(f"{name}: duplicate item {item}")
            seen.add(item)
            conf, note = notes.get(item, (default_conf, ""))
            rows.append([item, va, vb, SOURCE, conf, note])
    write_csv(OUT / f"{name}.csv", ["item", attr_a, attr_b, "source", "confidence", "notes"],
              rows)
    return len(rows)


def build_numbers() -> int:
    """Two-digit numbers 10-99 excluding 50: parity x magnitude (< 50 / > 50). Programmatic,
    so every attribute is certain."""
    rows = []
    for n in range(10, 100):
        if n == 50:
            continue
        rows.append([str(n), "even" if n % 2 == 0 else "odd", "small" if n < 50 else "large",
                     "programmatic", "high", ""])
    write_csv(OUT / "numbers.csv", ["item", "parity", "magnitude", "source", "confidence",
                                    "notes"], rows)
    return len(rows)


def main() -> None:
    print("categories:", build_categories())
    print("animals:", build_crossed("animals", ANIMALS, ANIMAL_NOTES, "class", "domestication"))
    print("countries:", build_crossed("countries", COUNTRIES, COUNTRY_NOTES, "continent",
                                      "coast"))
    print("words:", build_crossed("words", WORDS, WORD_NOTES, "category", "first_letter"))
    print("numbers:", build_numbers())


if __name__ == "__main__":
    main()
