from typing import Dict, Optional, List, Tuple


class GameState:
    """Holds play session state and flags."""

    def __init__(self, game_design: Dict):
        self.game_design = game_design

        self.current_node_id = "root"

        self.characters = {}

        print("Initializing new game state...")
        for char in game_design.get('characters', []):
            char_name = char.get('name')
            if char_name:
                self.characters[char_name] = {
                    "met": False,
                    "story_flags": []
                }

        self.story_flags = []
        self.choices_made = []

    def add_story_flag(self, flag: str):
        """Record a story flag if not already set."""
        if flag not in self.story_flags:
            self.story_flags.append(flag)
