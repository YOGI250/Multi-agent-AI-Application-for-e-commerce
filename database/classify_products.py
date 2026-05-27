# database/classify_products.py
# Run once to classify all products into product_type
# Usage: python3 -m database.classify_products

import logging
from database.connection import SessionLocal
from database.models import Product

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Classification rules — order matters, most specific first
# Each rule: (product_type, [keywords that must appear in name])
RULES = [
    # Cables — check before charger/usb since many names contain both
    ("cable",        ["usb cable", "type c cable", "type-c cable", "lightning cable",
                      "charging cable", "data cable", "hdmi cable", "braided cable",
                      "micro usb cable", "aux cable", "audio cable", "nylon cable"]),

    # Chargers
    ("charger",      ["charger", "charging adapter", "power adapter", "wall adapter",
                      "car charger", "fast charger", "wireless charger"]),

    # Mousepad — check before mouse so "gaming mousepad" doesn't match "gaming mouse"
    ("mousepad",     ["mouse pad", "mousepad", "gaming mousepad", "gaming pad",
                      "desk mat", "wrist rest"]),

    # Mouse
    ("mouse",        ["wireless mouse", "wired mouse", "gaming mouse", "optical mouse",
                      "bluetooth mouse", "silent mouse", "ergonomic mouse",
                      " mouse,", " mouse "]),

    # Keyboard
    ("keyboard",     ["keyboard", "mechanical keyboard", "wireless keyboard",
                      "bluetooth keyboard"]),

    # Headphones and earphones
    ("headphones",   ["headphone", "earphone", "earbud", "tws", "in-ear",
                      "over-ear", "neckband", "wireless earphone", "wired earphone",
                      "noise cancelling", "bluetooth headset"]),

    # Speakers
    ("speaker",      ["speaker", "soundbar", "bluetooth speaker", "portable speaker",
                      "home theatre", "subwoofer"]),

    # Laptop stand — check before laptop so accessories aren't misclassified
    ("stand",        ["laptop stand", "monitor stand", "phone stand", "tablet stand",
                      "adjustable stand", "foldable stand", "cooling stand",
                      "laptop cooling pad", "cooling pad", "laptop table",
                      "lapdesk", "lap desk", "bed table", "foldable table"]),

    # Laptop bag and sleeve — check before laptop
    ("laptop_bag",   ["laptop bag", "laptop sleeve", "laptop case", "laptop backpack",
                      "laptop pouch", "laptop cover"]),

    # Actual laptops — only match products that ARE laptops, not accessories for laptops.
    # Uses specific laptop product names and patterns.
    # "macbook", "chromebook", bare "laptop" intentionally excluded — these words appear
    # in compatibility text for accessories ("compatible with MacBook", "for Chromebook")
    # and cause misclassification. Use brand+model patterns instead.
    ("laptop",       ["gaming laptop", "notebook computer",
                      "thin & light laptop", "thin and light laptop",
                      "business laptop", "laptop computer",
                      "intel core i3 laptop", "intel core i5 laptop",
                      "intel core i7 laptop", "intel core i9 laptop",
                      "amd ryzen laptop", "ryzen 5 laptop", "ryzen 7 laptop",
                      "asus rog", "asus tuf gaming",
                      "msi gaming laptop", "msi laptop",
                      "hp pavilion laptop", "dell inspiron laptop",
                      "lenovo ideapad", "lenovo thinkpad", "lenovo legion",
                      "acer nitro", "acer aspire"]),

    # Tablet
    ("tablet",       ["tablet", "ipad", "android tablet", "fire tablet"]),

    # Smartwatch and fitness bands
    ("smartwatch",   ["smartwatch", "smart watch", "fitness band", "fitness tracker",
                      "activity tracker", "sport watch"]),

    # Monitor and display
    ("monitor",      ["monitor", "led monitor", "gaming monitor", "display screen",
                      "computer screen"]),

    # Webcam
    ("webcam",       ["webcam", "web camera", "hd camera", "usb camera"]),

    # Router and wifi
    ("router",       ["router", "wifi adapter", "wifi dongle", "wireless adapter",
                      "network adapter", "wi-fi adapter", "access point"]),

    # USB Hub
    ("usb_hub",      ["usb hub", "usb splitter", "type c hub", "port hub",
                      "multiport adapter"]),

    # Pen drive and storage
    ("pendrive",     ["pen drive", "pendrive", "flash drive", "usb drive",
                      "usb stick"]),

    # SSD
    ("ssd",          ["ssd", "solid state", "nvme", "m.2 drive"]),

    # Hard disk
    ("hard_disk",    ["hard disk", "hard drive", "hdd", "external drive",
                      "portable drive"]),

    # RAM
    ("ram",          ["ram", "ddr4", "ddr5", "laptop memory", "desktop memory",
                      "memory module"]),

    # Memory card
    ("memory_card",  ["memory card", "sd card", "microsd", "micro sd",
                      "card reader"]),

    # Printer
    ("printer",      ["printer", "inkjet", "laser printer", "all-in-one printer"]),


    # Phone case and accessories
    ("phone_case",   ["phone case", "back cover", "phone cover", "mobile case",
                      "screen protector", "tempered glass", "phone holder",
                      "mobile holder"]),

    # Extension board
    ("extension",    ["extension board", "extension cord", "surge protector",
                      "power strip", "multi-plug"]),

    # Fan
    ("fan",          ["ceiling fan", "table fan", "pedestal fan", "exhaust fan",
                      "tower fan", "wall fan", " fan "]),

    # Mixer and kitchen appliances
    ("mixer",        ["mixer", "blender", "juicer", "mixer grinder", "hand blender",
                      "food processor"]),

    # Iron
    ("iron",         ["steam iron", "dry iron", "clothes iron", "garment steamer",
                      " iron "]),

    # Kettle
    ("kettle",       ["electric kettle", "kettle", "water kettle"]),

    # Water heater and geyser
    ("water_heater", ["geyser", "water heater", "instant water heater",
                      "storage water heater"]),

    # Room heater
    ("room_heater",  ["room heater", "oil heater", "convector heater",
                      "fan heater", "infrared heater"]),

    # Vacuum cleaner
    ("vacuum",       ["vacuum cleaner", "robot vacuum", "robotic vacuum",
                      "wet and dry vacuum", "handheld vacuum"]),

    # Air purifier
    ("air_purifier", ["air purifier", "hepa filter", "air cleaner"]),

    # Water purifier
    ("water_purifier",["water purifier", "ro purifier", "uv purifier",
                       "water filter"]),

    # Camera
    ("camera",       ["digital camera", "action camera", "dslr", "mirrorless",
                      "security camera", "cctv", "dashcam", "dash cam"]),

    # Pen and stationery
    ("pen",          ["ball pen", "gel pen", "ballpoint", "fountain pen",
                      "ink pen", " pens", "pen set"]),

    # Notebook and diary
    ("notebook",     ["notebook", "diary", "journal", "spiral notebook",
                      "composition book"]),

    # LED bulb and lights
    ("light",        ["led bulb", "led light", "tube light", "smart bulb",
                      "string light", "night light", "desk lamp", "table lamp",
                      "floor lamp"]),

    # Trimmer and grooming
    ("trimmer",      ["trimmer", "shaver", "epilator", "hair clipper",
                      "beard trimmer", "body groomer"]),

    # Toothbrush
    ("toothbrush",   ["electric toothbrush", "toothbrush"]),

    # Microwave and cooking
    ("microwave",    ["microwave", "oven", "otg", "air fryer", "toaster"]),

    # Washing machine
    ("washing_machine", ["washing machine", "washer", "dryer"]),

    # Refrigerator
    ("refrigerator", ["refrigerator", "fridge", "freezer"]),

    # AC
    ("ac",           ["air conditioner", " ac ", "split ac", "window ac",
                      "inverter ac"]),
]


def classify_product(name: str) -> str:
    name_lower = name.lower()
    for product_type, keywords in RULES:
        for keyword in keywords:
            if keyword in name_lower:
                return product_type
    return "other"


def run():
    db = SessionLocal()
    try:
        products = db.query(Product).all()
        total    = len(products)
        logger.info(f"Classifying {total} products...")

        type_counts = {}
        for i, product in enumerate(products):
            ptype = classify_product(product.name)
            product.product_type = ptype
            type_counts[ptype] = type_counts.get(ptype, 0) + 1

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i+1}/{total}...")

        db.commit()
        logger.info("Classification complete.")
        logger.info("\nProduct type distribution:")
        for ptype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {count:4d}x  {ptype}")

    except Exception as e:
        logger.error(f"Error: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    run()
