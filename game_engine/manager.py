import pygame
import sys
from typing import Optional

from .config import SCREEN_WIDTH, SCREEN_HEIGHT, FPS
from .data import GameDataLoader, StoryParser
from .state import GameState
from .scenes import TitleScene, DialogueScene

class GameManager:
    """Main game loop and scene control."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("AI Visual Novel Engine")
        self.clock = pygame.time.Clock()
        self.running = True
        
        self.load_game_data()

        self.parsed_story = {}
        if self.story_text:
            self.parsed_story = StoryParser.parse_story(self.story_text)

        if self.game_design:
            self.game_state = GameState(self.game_design)
        else:
            print("No game design found; cannot start.")
            self.game_state = None

        self.current_scene = TitleScene(self)

    def load_game_data(self):
        print("Loading game data...")

        self.game_design = GameDataLoader.load_game_design()
        self.story_text = GameDataLoader.load_story()

        if self.game_design:
            print(f"Title: {self.game_design.get('title')}")
        if self.story_text:
            print(f"Story file length: {len(self.story_text)} characters")

    def get_character_id(self, name: str) -> Optional[str]:
        """Resolve a character name to id from the design."""
        if not self.game_state or not self.game_state.game_design:
            return None
        
        for char in self.game_state.game_design.get('characters', []):
            if char.get('name') == name:
                return char.get('id')
        return None

    def start_story(self):
        self.game_state.current_node_id = "root"
        self.play_current_scene()

    def on_scene_complete(self, scene_name: str):
        print(f"Scene finished: {scene_name}")
        print("Story block complete; back to title.")
        self.change_scene(TitleScene(self))

    def play_current_scene(self):
        state = self.game_state
        node_id = state.current_node_id

        if not node_id:
            print("current_node_id is empty")
            return

        if node_id not in self.parsed_story:
            print(f"No script for node: {node_id}")
            return

        lines = self.parsed_story[node_id]
        scene_name = f"Node: {node_id}"
        print(f"Playing node: {node_id} ({len(lines)} lines)")
        
        self.change_scene(DialogueScene(self, lines, scene_name))
    
    def change_scene(self, new_scene):
        self.current_scene = new_scene

    def run(self):
        print("\nGame started.")
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                self.current_scene.process_input(event)
            
            self.current_scene.update()
            self.current_scene.draw(self.screen)
            pygame.display.flip()
            self.clock.tick(FPS)
        
        pygame.quit()
        sys.exit()
