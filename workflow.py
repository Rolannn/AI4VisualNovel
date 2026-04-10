"""
Workflow Controller
~~~~~~~~~~~~~~~~~~~
Coordinates agent execution and manages the full lifecycle of game generation and runtime.
"""

import logging
import json
import os
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List
import time
from pathlib import Path
import re

from agents.producer_agent import ProducerAgent
from agents.designer_agent import DesignerAgent
from agents.artist_agent import ArtistAgent
from agents.writer_agent import WriterAgent
from agents.actor_agent import ActorAgent
from agents.config import PathConfig, APIConfig, WriterConfig, DesignerConfig, RAGConfig
from agents.story_graph import StoryGraph
from game_engine.data import StoryParser

# Constants
logger = logging.getLogger(__name__)


class WorkflowController:
    """Workflow controller - orchestrates all agents."""

    # It does not directly generate content; instead, it:

    # Orchestrates different Agents
    # Manages state (game_design)
    # Controls the sequence of the generation process
    
    def __init__(self):
        """Initialize workflow controller."""
        # Ensure log and data directories exist
        PathConfig.ensure_directories()
        
        self.producer = None
        self.designer = None
        self.artist = None
        self.writer = None
        self.actors = {}  # Store all actor agents: {name: ActorAgent}
        self.expressions_db = self._load_expressions()  # Expression library state
        # Default connection config for status/load paths without initialize_agents
        self.api_key = None
        self.base_url = None
        
        self.game_design = None
        
        logger.info(" Workflow controller initialized")
    
    def initialize_agents(
        self,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None
    ):
        """
        Initialize base agents (Producer, Artist, Writer).
        Actor agents are initialized after game design is loaded/generated.
        """
        logger.info(" Initializing agent system...")
        
        try:
            self.api_key = openai_api_key
            self.base_url = openai_base_url
            
            # Initialize Producer agent
            logger.info("    Initializing Producer agent (Reviewer)...")
            self.producer = ProducerAgent(api_key=openai_api_key, base_url=openai_base_url)

            # Initialize Designer agent
            logger.info("    Initializing Designer agent (Designer)...")
            self.designer = DesignerAgent(api_key=openai_api_key, base_url=openai_base_url)
            
            # Initialize Artist agent
            logger.info("    Initializing Artist agent...")
            self.artist = ArtistAgent(api_key=openai_api_key, base_url=openai_base_url)
            
            # Initialize Writer agent
            logger.info("     Initializing Writer agent...")
            self.writer = WriterAgent(api_key=openai_api_key, base_url=openai_base_url)
            
            logger.info(" Base agents initialized")
            
        except Exception as e:
            logger.error(f" Agent initialization failed: {e}")
            raise
    def _initialize_actors(self):
        """Initialize actor agents from game design."""
        if not self.game_design:
            raise ValueError("Game design is not loaded; cannot initialize actors")
            
        logger.info(" Initializing actor agents...")
        self.actors = {}
        for char_info in self.game_design.get('characters', []):
            name = char_info.get('name')
            if name:
                # Initialize expression library for this character
                self._initialize_character_expressions(name)
                
                actor = ActorAgent(
                    character_info=char_info,
                    api_key=self.api_key,
                    base_url=self.base_url
                )
                self.actors[name] = actor
                is_protagonist = char_info.get('is_protagonist', False)
                role_label = " (Protagonist)" if is_protagonist else ""
                logger.info(f"    Actor ready: {name}{role_label}")
    
    def create_new_game(
        self,
        character_count: int = 3,
        requirements: str = "",
        franchise: str = "",
        fan_characters: Optional[List[str]] = None,
        fan_docs_dir: str = "",
        rag_force_rebuild: bool = False,
        rag_language: str = "",
    ) -> Dict[str, Any]:
        """
        Create a new game (full flow: [RAG] -> design -> cast -> generate -> finalize).

        Args:
            character_count:    Number of characters (including protagonist)
            requirements:       User requirements text (from requirements_file)
            franchise:          Fan-fiction mode - franchise/IP name (e.g. "Genshin Impact")
            fan_characters:     Fan-fiction mode - list of required canon characters
            fan_docs_dir:       Fan-fiction mode - local supplemental docs directory (optional)
            rag_force_rebuild:  Fan-fiction mode - force rebuild knowledge base
            rag_language:       Fan-fiction mode - Wikipedia language (default from RAGConfig)
        """
        logger.info("="*60)
        logger.info(" Start creating a new game")
        logger.info("="*60)
        
        try:
            # ------------------------------------------------------
            # Step 0 (fan-fiction mode): build RAG knowledge base and augment requirements
            # ------------------------------------------------------
            if franchise:
                logger.info("\n[Step 0/6] Fan-fiction mode: building RAG knowledge base...")
                logger.info(f"   IP: {franchise}")
                logger.info(f"   Characters: {fan_characters or '(AI decides)'}")

                # Lazy import to avoid extra deps when fan-fiction mode is not used
                from agents.rag_agent import RAGAgent

                language = rag_language or RAGConfig.WIKIPEDIA_LANGUAGE
                force_rebuild = rag_force_rebuild or RAGConfig.FORCE_REBUILD

                rag_agent = RAGAgent(franchise=franchise, language=language)
                total_docs = rag_agent.build_knowledge_base(
                    franchise=franchise,
                    characters=fan_characters or [],
                    docs_dir=fan_docs_dir or None,
                    force_rebuild=force_rebuild,
                )

                logger.info(f"    Knowledge base has {total_docs} document chunks")
                logger.info("    Injecting franchise knowledge into creation requirements...")

                requirements = rag_agent.build_requirements_with_rag(
                    user_requirements=requirements,
                    franchise=franchise,
                    characters=fan_characters or [],
                )

                logger.info("    RAG knowledge injection complete")

                # # ------------------------------------------------------
                # # [DEBUG MODE] Print RAG retrieval results and return early without calling generation models
                # # TODO: Remove this block after testing (from "# [DEBUG MODE]" to "# [/DEBUG MODE]")
                # # ------------------------------------------------------
                # SEP = "=" * 80
                # print(f"\n{SEP}")
                # print(" RAG DEBUG MODE - Below are retrieval results; no generation model is called")
                # print(SEP)

                # # 1. Knowledge Base Stats
                # print("\n [Knowledge Base Stats]")
                # stats = rag_agent.get_stats()
                # for k, v in stats.items():
                #     print(f"   {k}: {v}")

                # # 2. Franchise overview retrieval results
                # print(f"\n{SEP}")
                # print(f" [Franchise Overview - {franchise}]")
                # print(SEP)
                # print(rag_agent.get_franchise_overview(n_results=3))

                # # 3. Per-character retrieval results (raw chunks)
                # for char_name in (fan_characters or []):
                #     print(f"\n{SEP}")
                #     print(f" [Character Retrieval - {char_name}]")
                #     print(SEP)

                #     # Low-level search: show raw chunks
                #     raw_results = rag_agent.kb.search(
                #         f"{char_name} character personality appearance background story",
                #         n_results=5,
                #     )
                #     if raw_results:
                #         for i, doc in enumerate(raw_results, 1):
                #             meta = doc.get("metadata", {})
                #             print(f"\n  --- Chunk {i} | section: {meta.get('section', '?')} | source: {meta.get('source', '?')} ---")
                #             print(f"  {doc['text'][:600]}")
                #     else:
                #         print("  (no result)")

                # # 4. Final injected requirements (truncated)
                # print(f"\n{SEP}")
                # print(" [Final requirements injected to DesignerAgent (first 3000 chars)]")
                # print(SEP)
                # print(requirements[:3000])
                # if len(requirements) > 3000:
                #     print(f"\n  ...(total {len(requirements)} chars, truncated)")

                # print(f"\n{SEP}")
                # print(" RAG DEBUG complete. To generate a game, comment out the [DEBUG MODE] block in workflow.py.")
                # print(SEP)
                # return {"rag_debug": True, "stats": stats, "requirements_preview": requirements[:500]}
                # # ------------------------------------------------------
                # # [/DEBUG MODE]
                # # ------------------------------------------------------

            # Step 1: check or generate game design document
            logger.info("\n[Step 1/6] Checking game design document...")
            existing_design = self.producer.load_game_design()
            if existing_design:
                logger.info(f" Found existing game design: {existing_design['title']}")
                self.game_design = existing_design
            else:
                logger.info("   No game design found, Designer starts drafting...")
                self.game_design = self.designer.generate_game_design(
                    character_count=character_count,
                    requirements=requirements
                )
                
                # Producer review loop (multi-iteration)
                max_iterations = 3
                current_iteration = 0
                
                while current_iteration < max_iterations:
                    logger.info(f"    Producer reviewing design draft (round {current_iteration + 1})...")
                    feedback = self.producer.critique_game_design(
                        self.game_design, 
                        requirements,
                        expected_nodes=self.designer.config.TOTAL_NODES,
                        expected_characters=character_count
                    )
                    
                    if feedback == "PASS":
                        logger.info("    Approved by Producer")
                        break
                    
                    logger.info(f"     Producer feedback: {feedback[:100]}...")
                    logger.info("    Designer is revising based on feedback...")
                    # Use unified interface: pass feedback and previous_game_design
                    self.game_design = self.designer.generate_game_design(
                        character_count=character_count,
                        requirements=requirements,
                        feedback=feedback,
                        previous_game_design=self.game_design
                    )
                    current_iteration += 1
                
                if current_iteration >= max_iterations:
                    logger.warning("    Reached max review iterations; Producer force-approves current version.")
                
                # Save final version
                self.producer.save_game_design(self.game_design)
            
            # Step 2: initialize actors
            logger.info("\n[Step 2/6] Initializing cast...")
            
            # Sync expression library with current design (remove stale characters)
            self._sync_expressions_with_design()
            
            self._initialize_actors()
            
            # Step 3: generate full story
            logger.info(f"\n[Step 3/6] Generating full story (DAG-based)...")
            self._generate_full_story()
            
            # Step 4: scan script and update expression library
            logger.info("\n[Step 4/6] Scanning script and syncing expression library...")
            self._scan_story_for_expressions()
            
            # Step 5: generate all art assets (backgrounds + sprites)
            logger.info("\n[Step 5/6] Generating art assets (backgrounds + character sprites)...")
            
            # 2. Generate scene backgrounds
            logger.info("    Generating scene backgrounds...")
            locations = [scene['name'] for scene in self.game_design.get('scenes', [])]
            self.artist.generate_all_backgrounds(
                locations,
                story_background=self.game_design.get('background'),
                art_style=self.game_design.get('art_style')
            )
            
            # 3. Generate all character sprites
            logger.info("    Generating all character sprites...")
            self._generate_character_assets()
            
            # Step 6: generate title screen (all art assets are now available)
            logger.info("\n[Step 6/6] Generating title screen...")
            character_ref_images = []
            for char_info in self.game_design.get('characters', []):
                char_id = char_info.get('id', char_info.get('name'))
                # Try finding neutral or other expressions
                char_dir = os.path.join(PathConfig.CHARACTERS_DIR, char_id)
                if os.path.exists(char_dir):
                    # Prefer neutral
                    neutral_path = os.path.join(char_dir, "neutral.png")
                    if os.path.exists(neutral_path):
                        character_ref_images.append(neutral_path)
                    else:
                        # Fallback to any png
                        try:
                            files = [f for f in os.listdir(char_dir) if f.endswith('.png')]
                            if files:
                                character_ref_images.append(os.path.join(char_dir, files[0]))
                        except OSError:
                            pass

            self.artist.generate_title_image(
                title=self.game_design.get('title', 'My Visual Novel'),
                background_desc=self.game_design.get('background', 'A romantic story'),
                character_images=character_ref_images
            )
            
            logger.info("\n" + "="*60)
            logger.info(" Game production complete")
            return self.game_design
            
        except Exception as e:
            logger.error(f" Game creation failed: {e}")
            raise

    def _generate_expression_with_critique(
        self, 
        actor: ActorAgent, 
        expression: str, 
        reference_image_path: Optional[str] = None,
        additional_feedback: str = ""
    ) -> Optional[str]:
        """
        Generate a single expression sprite with critique/retry loop
        
        Args:
            actor: Actor agent
            expression: expression name
            reference_image_path: reference image path (usually neutral expression)
            additional_feedback: extra guidance text (e.g. actor expression description)
            
        Returns:
            Returns image path on success, otherwise None
        """
        max_retries = 3
        current_try = 0
        feedback = additional_feedback
        previous_attempt_path = None  # Save image path from previous attempt
        
        while current_try < max_retries:
            logger.info(f"        Generating expression [{expression}] (attempt {current_try + 1}/{max_retries})...")
            
            # Build reference image list:
            # 1) Always include base reference (usually neutral) as a positive anchor
            # 2) Include previous failed attempt as additional context when available
            ref_paths = []
            if reference_image_path:
                ref_paths.append(reference_image_path)
            if previous_attempt_path:
                ref_paths.append(previous_attempt_path)
            
            # Generate image
            generated_paths = self.artist.generate_character_images(
                character=actor.character_info,
                expressions=[expression],
                feedback=feedback,
                reference_image_paths=ref_paths if ref_paths else None,
                story_background=self.game_design.get('background'),
                art_style=self.game_design.get('art_style')
            )
            
            image_path = generated_paths.get(expression)
            if not image_path:
                logger.warning(f"       Image generation failed")
                return None
                
            # Critique image
            critique_result = actor.critique_visual(
                image_path=image_path, 
                expression=expression,
                reference_image_path=reference_image_path,  # Use neutral as reference during critique
                story_background=self.game_design.get('background'),
                art_style=self.game_design.get('art_style')
            )
            
            if critique_result == "PASS":
                logger.info(f"       Critique passed: {expression}")
                return image_path
            else:
                logger.warning(f"        Critique failed: {critique_result[:100]}...")
                
                # Archive failed image and metadata to image_log (for analysis)
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    base_name = f"FAILED_{timestamp}_{actor.name}_{expression}"
                    
                    # 1) Archive image
                    fail_path = os.path.join(PathConfig.IMAGE_LOG_DIR, f"{base_name}.png")
                    shutil.copy2(image_path, fail_path)
                    
                    # 2) Archive metadata (character info, feedback, etc.)
                    meta_path = os.path.join(PathConfig.IMAGE_LOG_DIR, f"{base_name}.json")
                    meta_data = {
                        "timestamp": timestamp,
                        "character": actor.character_info,
                        "expression": expression,
                        "prompt_feedback": feedback,     # Feedback used for this attempt
                        "critique_result": critique_result, # Rejection reason from critique agent
                        "try_count": current_try + 1
                    }
                    with open(meta_path, 'w', encoding='utf-8') as f:
                        json.dump(meta_data, f, ensure_ascii=False, indent=4)
                        
                    logger.info(f"       Archived failed sample to: {PathConfig.IMAGE_LOG_DIR}")
                    previous_attempt_path = fail_path # Use as negative reference in next round
                except Exception as e:
                    logger.warning(f"        Archive failed: {e}")
                    previous_attempt_path = image_path
                
                current_try += 1
                
                # Merge critique into next-round feedback
                if additional_feedback:
                    feedback = f"{additional_feedback}\n\nPrevious critique: {critique_result}"
                else:
                    feedback = critique_result
                
                # Reached max retries; keep last generated image
                if current_try >= max_retries:
                    logger.warning(f"        Reached max retries; keep last generated image for gameplay")
                    return image_path
        
        return None
        
        return None



    def _generate_character_assets(self):
        """
        Generate all character sprite assets (with critique loop).
        Strategy:
        1. Generate neutral for all characters first (protagonist as style reference).
        2. Generate other expressions using each character's neutral as reference.
        """
        logger.info(" Generating character sprite assets...")
        
        # Phase 1: generate neutral expressions for all characters
        logger.info("    Phase 1: generate neutral expressions for all characters")
        
        style_reference_image = None
        
        # Generate protagonist neutral first as global style reference
        for char_name, actor in self.actors.items():
            if actor.character_info.get('is_protagonist', False):
                logger.info(f"       Generating protagonist {char_name} neutral...")
                
                char_id = actor.character_info.get('id', actor.name)
                char_dir = os.path.join(PathConfig.CHARACTERS_DIR, char_id)
                neutral_path = os.path.join(char_dir, "neutral.png")
                
                if os.path.exists(neutral_path):
                    logger.info(f"          Already exists; using as global style reference")
                    style_reference_image = neutral_path
                else:
                    logger.info(f"          Generating...")
                    neutral_path = self._generate_expression_with_critique(
                        actor=actor,
                        expression="neutral",
                        reference_image_path=None,
                        additional_feedback=""
                    )
                    if neutral_path:
                        style_reference_image = neutral_path
                        logger.info(f"          Generated; using as global style reference")
                    else:
                        logger.error(f"          Generation failed (API call failed)")
                
                break
        
        if not style_reference_image:
            logger.warning("        Protagonist neutral missing; other characters will be generated independently")
        
        # Generate neutral for other characters
        for char_name, actor in self.actors.items():
            if actor.character_info.get('is_protagonist', False):
                continue  # Protagonist already processed
            
            logger.info(f"       Generating {char_name} neutral...")
            
            char_id = actor.character_info.get('id', actor.name)
            char_dir = os.path.join(PathConfig.CHARACTERS_DIR, char_id)
            neutral_path = os.path.join(char_dir, "neutral.png")
            
            if os.path.exists(neutral_path):
                logger.info(f"          Already exists")
            else:
                logger.info(f"          Generating...")
                neutral_path = self._generate_expression_with_critique(
                    actor=actor,
                    expression="neutral",
                    reference_image_path=style_reference_image,
                    additional_feedback="Match the art style of the protagonist." if style_reference_image else ""
                )
                
                if neutral_path:
                    logger.info(f"          Generated successfully")
                else:
                    logger.error(f"          Generation failed (API call failed)")
        
        # Phase 2: generate all other expressions
        logger.info("    Phase 2: generate all other expressions")
        
        for char_name, actor in self.actors.items():
            char_id = actor.character_info.get('id', actor.name)
            char_dir = os.path.join(PathConfig.CHARACTERS_DIR, char_id)
            
            # Get all registered expressions for this character
            expressions = self._get_character_expressions(char_name)
            
            # Use neutral as reference
            neutral_path = os.path.join(char_dir, "neutral.png")
            ref_path = neutral_path if os.path.exists(neutral_path) else None
            
            # Filter non-neutral expressions
            other_expressions = [e for e in expressions if e != "neutral"]
            
            if not other_expressions:
                continue
            
            logger.info(f"       Generating other expressions for {char_name}: {other_expressions}")
            
            for expr in other_expressions:
                # Check if file already exists
                img_path = os.path.join(char_dir, f"{expr}.png")
                if os.path.exists(img_path):
                    logger.info(f"          {expr} Already exists")
                    continue
                
                logger.info(f"          Generating {expr}...")
                
                # Let Actor describe this expression
                description = actor.generate_expression_description(expr)
                additional_feedback = f"Expression description: {description}"
                
                # Generate with critique loop
                result_path = self._generate_expression_with_critique(
                    actor=actor,
                    expression=expr,
                    reference_image_path=ref_path,
                    additional_feedback=additional_feedback
                )
                
                if result_path:
                    logger.info(f"          {expr} Generated successfully")
                else:
                    logger.error(f"          {expr} generation failed")

    def _scan_story_for_expressions(self):
        """
        Scan story.txt, extract character expression tags, and update expression library.
        Ensure all expressions used in script are recorded.
        """
        logger.info(" Scanning script file for expression usage...")
        
        story_path = PathConfig.STORY_FILE
        if not os.path.exists(story_path):
            logger.warning("    story.txt not found, skipping scan")
            return
        
        try:
            with open(story_path, 'r', encoding='utf-8') as f:
                story_content = f.read()
            
            # Extract all <image id="CharacterName">expression</image> tags
            # Regex supports non-English character names
            import re
            pattern = r'<image\s+id="([^"]+)">([^<]+)</image>'
            matches = re.findall(pattern, story_content)
            
            if not matches:
                logger.info("   No character expression tags found in script")
                return
            
            # Count expressions used per character
            character_expressions_in_story = {}
            for char_name, expression in matches:
                char_name = char_name.strip()
                expression = expression.strip()
                
                if char_name not in character_expressions_in_story:
                    character_expressions_in_story[char_name] = set()
                character_expressions_in_story[char_name].add(expression)
            
            # Update expression library
            updated_count = 0
            for char_name, expressions in character_expressions_in_story.items():
                # Check whether this character exists in actor list
                if char_name not in self.actors:
                    logger.warning(f"    Unknown character in script: {char_name}, skipping")
                    continue
                
                # Add new expressions
                added = self._add_expressions_to_character(char_name, list(expressions))
                if added:
                    logger.info(f"    New expressions for {char_name}: {added}")
                    updated_count += len(added)
            
            if updated_count > 0:
                logger.info(f"    Expression library updated, added {updated_count} expressions")
            else:
                logger.info("    Expression library already up to date")
                
        except Exception as e:
            logger.error(f"    Failed to scan script: {e}")

    def _generate_full_story(self):
        """Generate full story (supports tree and DAG structures)."""
        try:
            # Create story graph object (auto-compatible with tree and DAG)
            story_graph = StoryGraph(self.game_design)
            
            # Validate graph structure
            is_valid, error_msg = story_graph.validate()
            if not is_valid:
                logger.error(f" Story graph validation failed: {error_msg}")
                return
            
            # Use topological sorting to determine generation order
            node_order = story_graph.topological_sort()
            logger.info(f" Story graph contains {len(node_order)} nodes")
            
            # Node summary and content cache
            node_summaries = {}
            node_contents = {}
            
            for idx, node_id in enumerate(node_order, 1):
                node_info = story_graph.get_node(node_id)
                logger.info(f"\n [{idx}/{len(node_order)}] Generating node: {node_id}")
                
                # Check whether node content already exists
                story_path = Path(PathConfig.STORY_FILE)
                node_exists = False
                if story_path.exists():
                    with open(story_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if f"=== Node: {node_id} ===" in content:
                            node_exists = True
                            logger.info(f"   Node script already exists, skipping generation")
                            
                            # Extract content for context
                            pattern = f"=== Node: {node_id} ===(.*?)(=== Node|$)"
                            match = re.search(pattern, content, re.DOTALL)
                            if match:
                                node_content = match.group(1).strip()
                                node_contents[node_id] = node_content
                                if node_id not in node_summaries:
                                    node_summary = self.writer.summarize_story(node_content)
                                    node_summaries[node_id] = node_summary
                
                if not node_exists:
                    # Build context (supports multiple parents)
                    parents = story_graph.get_parents(node_id)
                    
                    # Long-term memory: ancestor node summaries
                    long_term_memory = self._build_long_term_memory(
                        node_id, story_graph, node_summaries
                    )
                    
                    # Short-term memory: full content of direct parent nodes
                    short_term_memory = ""
                    if parents:
                        if len(parents) > 1:
                            # Merge node: tell LLM multiple paths have converged
                            parent_summaries = [
                                f"[Path {i+1}] {node_summaries.get(p, '(no summary)')}" 
                                for i, p in enumerate(parents) if p in node_summaries
                            ]
                            short_term_memory = (
                                "Multiple story paths converge here. Continue from shared memory:\n" + 
                                "\n".join(parent_summaries)
                            )
                        else:
                            # Normal node: use full parent node content
                            parent_contents = [node_contents.get(p, "") for p in parents if p in node_contents]
                            short_term_memory = "\n\n".join(parent_contents)
                    
                    full_context = f"{long_term_memory}\n\n[Recent story]:\n{short_term_memory}"
                    
                    # Generate script
                    node_performance_data = []
                    plot_summary = node_info.get('summary', '')
                    
                    # Identify present characters
                    char_names = list(self.actors.keys())
                    present_actors = list(self.actors.items())  # Use all characters directly
                    
                    if present_actors:
                        # ==================== Step 1: split into plot segments ====================
                        logger.info(f"  Splitting node {node_id} into plot segments...")
                        available_scenes = [scene['name'] for scene in self.game_design.get('scenes', [])]
                        # Get full character metadata
                        available_characters = self.game_design.get('characters', [])
                        
                        plots = self.writer.split_node_into_plots(
                            node_summary=plot_summary,
                            long_term_memory=long_term_memory,
                            available_scenes=available_scenes,
                            available_characters=available_characters,
                            segment_count=DesignerConfig.PLOT_SEGMENTS_PER_NODE
                        )
                        
                        if not plots:
                            logger.warning(f" Failed to split node {node_id} into plots, using original summary")
                            plots = [{"id": 1, "summary": plot_summary}]
                        
                        logger.info(f" Split into {len(plots)} segments")
                        
                        # ==================== Step 2: performance loop per segment ====================
                        all_plot_contexts = []
                        performance_log_path = os.path.join(PathConfig.TEXT_LOG_DIR, f"performance_{node_id}.jsonl")
                        
                        # Clear old performance log (if exists) so regeneration overwrites it
                        if os.path.exists(performance_log_path):
                            os.remove(performance_log_path)
                            logger.info(f"  Cleared old performance log for node {node_id}")
                        
                        for plot_idx, plot_info in enumerate(plots, 1):
                            plot_id = plot_info.get('id', plot_idx)
                            current_plot_summary = plot_info.get('summary', plot_summary)
                            
                            logger.info(f" Executing segment {plot_idx}/{len(plots)}: {current_plot_summary[:50]}...")
                            
                            # Dialogue accumulation buffer for this segment
                            plot_current_context = ""
                            turn_count = 0
                            safety_limit = 50  # Fixed safety cap: 50 turns
                            speaker_retry_count = 0  # Director retry counter
                            max_speaker_retries = 3
                            
                            # Build segment context: global history + previous segments
                            previous_plots_context = "\n\n".join(all_plot_contexts) if all_plot_contexts else ""
                            plot_full_context = full_context
                            if previous_plots_context:
                                plot_full_context += f"\n\n[Previous segments]:\n{previous_plots_context}"
                            
                            while turn_count < safety_limit:
                                current_total_context = f"{plot_full_context}\n\n[Current segment dialogue]:\n{plot_current_context}"
                                
                                present_char_names = [name for name, _ in present_actors]
                                # Extract character info dicts from ActorAgent
                                present_char_info = [actor.character_info for _, actor in present_actors]
                                next_speaker_name, plot_guidance = self.writer.decide_next_speaker(
                                    plot_summary=current_plot_summary,
                                    characters=present_char_info,
                                    story_context=current_total_context
                                )
                                
                                if "STOP" in next_speaker_name:
                                    logger.info(f" Director calls STOP for segment {plot_idx}")
                                    break
                                
                                next_actor = None
                                next_char_name = ""
                                for name, agent in present_actors:
                                    if name in next_speaker_name or next_speaker_name in name:
                                        next_actor = agent
                                        next_char_name = name
                                        break
                                
                                if not next_actor:
                                    speaker_retry_count += 1
                                    logger.warning(f" Director selected unknown character: {next_speaker_name}; retry speaker selection ({speaker_retry_count}/{max_speaker_retries})...")
                                    if speaker_retry_count < max_speaker_retries:
                                        # Continue loop and let director decide again
                                        continue
                                    else:
                                        logger.warning(f" Director failed to pick a valid character after {max_speaker_retries} retries; ending this segment dialogue")
                                        break
                                
                                # Valid character selected, reset retry counter
                                speaker_retry_count = 0
                                
                                # Build full metadata for other characters
                                other_chars = [
                                    actor.character_info for char_name, actor in present_actors
                                    if char_name != next_char_name
                                ]
                                available_expressions = self._get_expressions_str(next_char_name)
                                
                                enhanced_plot_summary = current_plot_summary
                                if plot_guidance:
                                    enhanced_plot_summary += f"\n[Director guidance]{plot_guidance}"
                                
                                performance = next_actor.perform_plot(
                                    plot_summary=enhanced_plot_summary,
                                    other_characters=other_chars,
                                    story_context=current_total_context,
                                    character_expressions=available_expressions
                                )
                                
                                # Log actor performance
                                self._log_performance(performance_log_path, {
                                    "node_id": node_id,
                                    "plot_id": plot_id,
                                    "character": next_char_name,
                                    "content": performance
                                })
                                
                                if performance.strip():
                                    plot_current_context += f"{performance}\n"
                                    self._update_character_expressions(next_char_name, performance)
                                    turn_count += 1
                                else:
                                    break
                            
                            all_plot_contexts.append(plot_current_context)
                            logger.info(f" Segment {plot_idx} complete ({turn_count} dialogue turns)")
                        
                        # ==================== Step 3: merge all segments into final script ====================
                        logger.info(f"  Writer is merging all segments for node {node_id}...")
                        
                        # Full dialogue from all segments
                        current_context = "\n\n".join(all_plot_contexts)
                        
                        # Get choice information
                        children = story_graph.get_children(node_id)
                        choices_data = [{"target": child_id, "text": choice_text} for child_id, choice_text in children]
                        
                        # Let writer polish and synthesize
                        polished_script = self.writer.synthesize_script(
                            plot_performances=[{"content": current_context}],
                            choices=choices_data,
                            story_context=full_context,
                            available_scenes=available_scenes,
                            available_characters=available_characters
                        )
                        
                        # Save polished script
                        self._save_node_story(node_id, polished_script)
                        node_contents[node_id] = polished_script
                        node_summary = self.writer.summarize_story(polished_script)
                        node_summaries[node_id] = node_summary
                    
                    logger.info(f" Node {node_id} script generation complete")
            
            logger.info("\n Full story generation complete")
            
        except Exception as e:
            logger.error(f" Story generation failed: {e}", exc_info=True)

    def load_existing_game(self) -> bool:
        """Load existing game data."""
        try:
            # If producer is not initialized, initialize it first
            if not self.producer:
                self.producer = ProducerAgent()
                
            self.game_design = self.producer.load_game_design()
            if not self.game_design:
                return False
                
            # Initialize actors
            self._initialize_actors()
            
            return True
        except Exception as e:
            logger.error(f" Failed to load game: {e}")
            return False

    def _load_expressions(self) -> Dict[str, List[str]]:
        """Load existing expression library."""
        expr_file = os.path.join(PathConfig.DATA_DIR, "character_expressions.json")
        if os.path.exists(expr_file):
            try:
                with open(expr_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f" Failed to load expression library: {e}; creating a new one")
                return {}
        return {}

    def _save_expressions(self):
        """Save expression library to file."""
        expr_file = os.path.join(PathConfig.DATA_DIR, "character_expressions.json")
        try:
            with open(expr_file, 'w', encoding='utf-8') as f:
                json.dump(self.expressions_db, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f" Failed to save expression library: {e}")

    def _get_character_expressions(self, character_name: str) -> List[str]:
        """Get existing expression library for a character."""
        return self.expressions_db.get(character_name, [])
    
    def _add_expressions_to_character(self, character_name: str, expressions: List[str]) -> List[str]:
        """
        Add expressions to a character expression library
        
        Args:
            character_name: Character name
            expressions: Expressions to add
            
        Returns:
            Newly added expressions
        """
        current_expressions = set(self._get_character_expressions(character_name))
        new_expressions = [expr for expr in expressions if expr not in current_expressions]
        
        if new_expressions:
            if character_name not in self.expressions_db:
                self.expressions_db[character_name] = []
            
            self.expressions_db[character_name].extend(new_expressions)
            self.expressions_db[character_name] = list(set(self.expressions_db[character_name]))  # Deduplicate
            self._save_expressions()
            
        return new_expressions

    def _update_character_expressions(self, character_name: str, text: str) -> List[str]:
        """Extract expression tags from text and update character expression library."""
        # Extract all <image id="name">expression</image> tags
        pattern = rf'<image\s+id="{re.escape(character_name)}">([^<]+)</image>'
        extracted_expressions = list(set(re.findall(pattern, text)))
        
        if not extracted_expressions:
            return []
        
        added = self._add_expressions_to_character(character_name, extracted_expressions)
        if added:
            logger.info(f" New expressions for {character_name}: {added}")
        return added

    def _initialize_character_expressions(self, character_name: str):
        """Initialize character expression library."""
        if character_name not in self.expressions_db:
            from agents.config import STANDARD_EXPRESSIONS
            initial_expressions = STANDARD_EXPRESSIONS.copy()
            self.expressions_db[character_name] = initial_expressions
            self._save_expressions()
            logger.info(f" Initialized expression library for {character_name}: {initial_expressions}")

    def _get_expressions_str(self, character_name: str) -> str:
        """Get comma-separated string of character expressions."""
        expressions = self._get_character_expressions(character_name)
        if not expressions:
            return "neutral, happy, sad, angry, surprised, shy"
        return ", ".join(expressions)

    def _sync_expressions_with_design(self):
        """Ensure expression library is synced with current game design."""
        if not self.game_design:
            return
            
        current_character_names = set(c.get('name') for c in self.game_design.get('characters', []) if c.get('name'))
        existing_character_names = set(self.expressions_db.keys())
        
        # Find characters to remove
        to_remove = existing_character_names - current_character_names
        
        if to_remove:
            logger.info(f" Sync expression library: remove stale characters {list(to_remove)}")
            for name in to_remove:
                del self.expressions_db[name]
            self._save_expressions()

    def get_game_status(self) -> Dict[str, Any]:
        """
        Get current game status.
        
        Returns:
            Game status dictionary
        """
        if not self.game_design:
            return {"initialized": False}
        
        # Count generated nodes
        total_nodes = len(self.game_design.get("story_graph", {}).get("nodes", {}))
        completed_nodes = 0
        
        story_path = Path(PathConfig.STORY_FILE)
        if story_path.exists():
            with open(story_path, "r", encoding="utf-8") as f:
                content = f.read()
                for node_id in self.game_design.get("story_graph", {}).get("nodes", {}):
                    if f"=== Node: {node_id} ===" in content:
                        completed_nodes += 1
        
        return {
            "initialized": True,
            "title": self.game_design.get('title', 'Unknown'),
            "completed_nodes": completed_nodes,
            "total_nodes": total_nodes
        }
    
    def _build_long_term_memory(
        self, 
        node_id: str, 
        story_graph: 'StoryGraph', 
        node_summaries: Dict[str, str]
    ) -> str:
        """
        Build long-term context from ancestor node summaries.
        
        For merge nodes: only use common ancestors to avoid conflicting branches.
        
        Args:
            node_id: Current node ID
            story_graph: Story graph object
            node_summaries: Node summary cache
            
        Returns:
            Context text
        """
        parents = story_graph.get_parents(node_id)
        
        # If this is a merge node (multiple parents)
        if len(parents) > 1:
            # Find common ancestors across all parents
            ancestor_sets = []
            for parent in parents:
                ancestors = self._get_ancestors(parent, story_graph)
                ancestor_sets.append(ancestors)
            
            # Intersect sets: keep nodes shared by all paths
            if ancestor_sets:
                common_ancestors = set.intersection(*ancestor_sets)
            else:
                common_ancestors = set()
            
            # Build context (sorted by ID)
            context_parts = []
            for ancestor_id in sorted(common_ancestors):
                if ancestor_id in node_summaries:
                    context_parts.append(f"Node {ancestor_id}: {node_summaries[ancestor_id]}")
            
            if context_parts:
                return "\n".join(context_parts)
            else:
                return "Story starts (multiple paths converge here)."
        
        # Normal node: use all ancestors
        else:
            ancestors = set()
            queue = parents.copy()
            
            while queue:
                parent = queue.pop(0)
                if parent not in ancestors:
                    ancestors.add(parent)
                    queue.extend(story_graph.get_parents(parent))
            
            # Build context
            context_parts = []
            for ancestor_id in sorted(ancestors):
                if ancestor_id in node_summaries:
                    context_parts.append(f"Node {ancestor_id}: {node_summaries[ancestor_id]}")
            
            return "\n".join(context_parts) if context_parts else "Game starts."
    
    def _get_ancestors(self, node_id: str, story_graph: 'StoryGraph') -> set:
        """Get all ancestors of a node (BFS)."""
        ancestors = set()
        queue = [node_id]
        
        while queue:
            current = queue.pop(0)
            if current not in ancestors:
                ancestors.add(current)
                queue.extend(story_graph.get_parents(current))
        
        return ancestors
    
    def _save_node_story(self, node_id: str, content: str):
        """
        Save node script to file.
        
        Args:
            node_id: Node ID
            content: Script content
        """
        story_path = Path(PathConfig.STORY_FILE)
        with open(story_path, 'a', encoding='utf-8') as f:
            f.write(f"\n=== Node: {node_id} ===\n")
            f.write(content)
            f.write("\n")
    
    def _append_choices_to_story(self, node_id: str, children: List[tuple]):
        """
        Append choices to the story file.
        
        Args:
            node_id: Current node ID
            children: [(child_id, choice_text), ...]
        """
        story_path = Path(PathConfig.STORY_FILE)
        with open(story_path, 'a', encoding='utf-8') as f:
            f.write("\n[CHOICES]\n")
            for idx, (child_id, choice_text) in enumerate(children, 1):
                if choice_text:
                    f.write(f'<choice target="{child_id}">{choice_text}</choice>\n')
                else:
                    # No explicit choice text; generate default option (normally unexpected in multi-choice)
                    default_text = f"Option {idx}"
                    logger.warning(f" Child node {child_id} of node {node_id} is missing choice_text; using default: {default_text}")
                    f.write(f'<choice target="{child_id}">{default_text}</choice>\n')
            f.write("\n")
    

    def _log_performance(self, log_path: str, data: dict):
        """
        Log performance process to a JSONL file.
        
        Args:
            log_path: Log file path
            data: Data dict to record
        """
        try:
            import json
            from datetime import datetime
            
            # Add concise timestamp (HH:MM:SS)
            data['timestamp'] = datetime.now().strftime("%H:%M:%S")
            
            # Append to jsonl file
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.warning(f" Failed to log performance: {e}")
    
    def _character_mentioned_in(self, char_name: str, text: str) -> bool:
        """Check whether a character is mentioned in text."""
        return char_name in text or char_name.lower() in text.lower()
