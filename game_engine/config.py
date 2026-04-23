import os
from pathlib import Path

# Display and timing
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 768
FPS = 60


class Colors:
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    BG_MORNING = (135, 206, 250)
    BG_AFTERNOON = (255, 160, 122)
    BG_EVENING = (25, 25, 112)

    UI_PANEL_BG = (30, 30, 40, 230)
    UI_BORDER = (255, 255, 255)
    UI_TEXT = (240, 240, 240)
    UI_TEXT_HIGHLIGHT = (255, 215, 0)

    BTN_NORMAL = (70, 130, 180)
    BTN_HOVER = (100, 149, 237)
    BTN_TEXT = (255, 255, 255)

    MAP_GRASS = (154, 205, 50)
    MAP_ROAD = (160, 160, 160)
    MAP_LAKE = (100, 149, 237)

    CHAR_ME = (100, 149, 237)
    CHAR_GIRL = (255, 105, 180)
    CHAR_MYSTERY = (138, 43, 226)
    CHAR_UNKNOWN = (169, 169, 169)


class DataPaths:
    import sys
    if getattr(sys, 'frozen', False):
        BASE_DIR = Path(sys._MEIPASS)
    else:
        BASE_DIR = Path(__file__).parent.parent

    DATA_DIR = BASE_DIR / "data"

    GAME_DESIGN_FILE = DATA_DIR / "game_design.json"
    STORY_FILE = DATA_DIR / "story.txt"

    IMAGES_DIR = DATA_DIR / "images"
    BACKGROUNDS_DIR = IMAGES_DIR / "backgrounds"
    CHARACTERS_DIR = IMAGES_DIR / "characters"
