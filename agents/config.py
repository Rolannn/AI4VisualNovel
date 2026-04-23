"""
Agent Configuration
~~~~~~~~~~~~~~~~~~~
Configuration, API Keys, model parameters and prompt templates for all Agents
"""

import os
from typing import Dict, Any
from pathlib import Path

# Load .env file
try:
    from dotenv import load_dotenv
    # Find .env file (in project root)
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment variables: {env_path}")
    else:
        print(f".env file not found: {env_path}")
except ImportError:
    print("python-dotenv not installed, cannot auto-load .env file")
    print("   Please run: pip install python-dotenv")

# ==================== API Configuration ====================
class APIConfig:
    """API key configuration"""
    # Provider configuration
    TEXT_PROVIDER = os.getenv("TEXT_PROVIDER", "google")
    IMAGE_PROVIDER = os.getenv("IMAGE_PROVIDER", "google")
    
    # OpenAI API (for GPT-4 and image generation)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    # Google Gemini API
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GOOGLE_BASE_URL = os.getenv("GOOGLE_BASE_URL", "")

    # Model name
    MODEL = os.getenv("MODEL", "gemini-3-pro-preview")
    
    # Image generation model
    IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-1.5") 


# ==================== Global Constants ====================
# Standard expression list
STANDARD_EXPRESSIONS = os.getenv("GAME_CHARACTER_EXPRESSIONS", "neutral").split(",")

# ==================== Designer Agent Configuration ====================
class DesignerConfig:
    """Designer Agent - responsible for drafting the game design document"""
    
    # Game content configuration
    TOTAL_NODES = int(os.getenv("GAME_TOTAL_NODES", "12"))
    DEFAULT_CHARACTER_COUNT = int(os.getenv("GAME_CHARACTER_COUNT", "3"))
    PLOT_SEGMENTS_PER_NODE = int(os.getenv("PLOT_SEGMENTS_PER_NODE", "3"))
    MAX_TURNS_PER_SEGMENT  = int(os.getenv("MAX_TURNS_PER_SEGMENT",  "20"))

    SYSTEM_PROMPT = """You are a senior Visual Novel designer, skilled at crafting engaging stories.
Your task is to design a complete Visual Novel game document using a **Directed Acyclic Graph (DAG) structure**, allowing different branches to diverge and converge, ensuring all characters have compelling roles across the complex story network.
"""


    GAME_DESIGN_PROMPT = """Please create a Visual Novel game design document.

Character count: {character_count} (including protagonist)
Story structure: Directed Acyclic Graph (DAG), supports multiple branches and path merges

[User Requirements]
{requirements}
(If the user requirements are empty, feel free to create freely; if there is content, please strictly follow the user requirements.)

[Important Rules]
1. Output a valid JSON object directly, without any other text
2. Do not use comments (do not write //)
3. Avoid using newline characters in all text content; use spaces instead
4. If quotes are needed in text, use single quotes or omit them
5. Ensure all strings are properly closed

[JSON Format Example]
{{
  "title": "Game Title",
  "background": "Background story and content introduction, 200-300 words",
  "art_style": "Art style description (e.g., Japanese anime, cyberpunk, steampunk, watercolor, pixel art, realistic, etc.), including color palette, atmosphere, and visual characteristics",
  "story_graph": {{
    "nodes": {{
      "root": {{
        "id": "root",
        "summary": "Neo-Tokyo 2045. The protagonist, a street hacker, finds an encrypted chip containing government scandal data in an abandoned subway station. As they prepare to leave, the cold-blooded agent Su Ya surrounds the platform with her team. In a hail of gunfire and electronic jamming, the protagonist barely escapes using knowledge of the terrain, but this means they are now a wanted criminal citywide. The 'Purification Plan' mentioned in the chip seems to foretell a devastating strike on the slums.",
        "type": "normal"
      }},
      "node1": {{
        "id": "node1",
        "summary": "After retreating to a safe house, the protagonist contacts Lin Yu, a former partner and genius programmer. While analyzing the chip's edge code, Lin Yu discovers a hidden set of physical coordinates pointing to an abandoned radar station in the suburbs. Meanwhile, the safe house defense system alerts that Su Ya's hunting squad is conducting a floor-by-floor sweep. The protagonist must decide: risk breaking out to the radar station for the truth, or attempt to infiltrate the agency's data network from the inside to dismantle the tracking.",
        "type": "normal"
      }},
      "node2": {{
        "id": "node2",
        "summary": "The protagonist drives to the radar station and finds an injured Lin Yu being cornered by unidentified attackers. After a daring rescue, Lin Yu reveals a backup card showing that Su Ya is actually an opponent of the 'Purification Plan'. Su Ya arrives alone and, instead of firing, whispers that the real mastermind is hiding in a cloud control room, and she needs the protagonist's hacking skills to disable the self-destruct sequence.",
        "type": "normal"
      }},
      "node3": {{
        "id": "node3",
        "summary": "The protagonist chooses the riskier path and reverse-infiltrates the agency's network. In a virtual confrontation, they discover Su Ya's action logs have been massively altered. Successfully stealing the control room's access key, the protagonist discovers that Lin Yu has already betrayed them, selling their location to the military in exchange for a pardon. In the rainy night streets, under Su Ya's covert protection, the protagonist barely escapes to the final showdown — the Central Tower.",
        "type": "normal"
      }},
      "node4": {{
        "id": "node4",
        "summary": "All clues converge at the cloud Central Tower. Whether it is Lin Yu's betrayal or Su Ya's cover, everything is now on the table. The protagonist stands before the control terminal as Su Ya and Lin Yu each argue their case. Lin Yu claims only the 'Purification' can restore order, while Su Ya insists on revealing the truth. The alarms outside are deafening; the delete key that will decide the city's fate is in the protagonist's hands. This is a final moment of sacrifice, betrayal, and redemption.",
        "type": "merge"
      }},
      "node5": {{
        "id": "node5",
        "summary": "The truth is made public, and the old order collapses in flames. The protagonist and Su Ya vanish into the chaotic crowd, becoming ghosts forgotten by history — but the city welcomes a long-awaited new dawn.",
        "type": "normal"
      }},
      "node6": {{
        "id": "node6",
        "summary": "The protagonist chooses compromise, gaining the false tranquility of the upper class. Lin Yu becomes the new hero, while Su Ya disappears into the shadows. Every night, the protagonist hears the laments echoing from the depths below.",
        "type": "normal"
      }}
    }},
    "edges": [
      {{"from": "root", "to": "node1", "choice_text": null}},
      {{"from": "node1", "to": "node2", "choice_text": "Head to the radar station"}},
      {{"from": "node1", "to": "node3", "choice_text": "Infiltrate the network"}},
      {{"from": "node2", "to": "node4", "choice_text": null}},
      {{"from": "node3", "to": "node4", "choice_text": null}},
      {{"from": "node4", "to": "node5", "choice_text": "Reveal the truth"}},
      {{"from": "node4", "to": "node6", "choice_text": "Accept the false peace"}}
    ]
    
    **[choice_text Rules]**:
    - **choice_text is null**: auto-advance, no player choice shown (for single-path natural continuation)
    - **choice_text has value**: must be concise, immersive option text (within 10 characters)
    - **No meta-tags**: do not add meta-tags to options, such as "(Technical Route)", "[Force Route]", "[BRANCH A]", etc.
    - **Bad examples**: "Contact hacker Jinx (Technical Route)", "[Rational Choice] Seek help"
    - **Good examples**: "Contact hacker Jinx", "Seek help", "Investigate alone"
  }},
  "characters": [
    {{
      "id": "english_identifier",
      "name": "Character Name",
      "gender": "gender",
      "is_protagonist": true,
      "personality": "Detailed personality description",
      "appearance": "Detailed appearance for AI image generation",
      "background": "Detailed backstory for a well-rounded character"
    }}
  ],
  "scenes": [
    {{
      "id": "scene_001",
      "name": "My Room",
      "description": "Detailed scene description, e.g.: The protagonist's bedroom, warmly furnished with a desk, bed, and personal items reflecting the protagonist's personality and interests.",
      "atmosphere": "Warm and peaceful"
    }}
  ]
}}

[Important Notes]:
1. The characters array must contain {character_count} characters, **including exactly 1 protagonist (is_protagonist: true)**
   - **Character name convention: use only one language for all character names, do not add parenthetical annotations**
   - Bad examples: "Ren (Ren)", "Alice (Alice)"
   - Good examples: "Ren", "Alice"
2. The scenes array must contain at least 5-10 different scenes
3. **story_graph must be a valid DAG**:
   - Each node must have a unique ID
   - from/to in edges must reference existing nodes
   - No cycles (circular references)
4. **choice_text rules (important!)**:
   - **null value**: auto-advance, no player choice needed (single-path continuation)
   - **non-null value**: must be concise, immersive option text (5-10 characters), describing the action directly
   - **No meta-tags**: no parenthetical comments, route identifiers, branch markers, or other immersion-breaking content
   - Bad examples: "Contact Jinx (Technical Route)", "[Rational] Seek help"
   - Good examples: "Contact Jinx", "Seek help", "Investigate alone"
5. **Node ID naming convention**: use uniform node + number format
   - Recommended format: "root", "node1", "node2", "node3", "node4" ...
   - Root node ID must be "root"
   - Do not use character names, Chinese, or non-uniform naming like n1, merge1
6. **Node types**:
   - **normal**: regular node (including start, middle, and ending nodes)
   - **merge**: convergence point (multiple paths merge here)
   - Note: the starting node is root; ending nodes are those with no outgoing edges
7. **Recommended structure**:
   - **STRICT REQUIREMENT: Total nodes must be EXACTLY {total_nodes} (±1 maximum). Do NOT exceed this limit under any circumstances.**
   - Provide multiple endings to increase replayability
8. **Path design and narrative coherence (critical!)**:
   - **Merge points must be logically connected**: clues, events, or character decisions from each branch should converge naturally at the merge point, such as: (1) two clues pointing to the same target; (2) different characters gathering at a key moment; (3) multiple events triggering the same consequence; (4) players forced to reunite at a location, etc.
   - **Merge node summary must be unified**: the merge node summary should not describe conditional cases, but rather a single unified event or scene.
   - **Different branches must not be unrelated**: for example, node2 "discover art room clue" and node3 "confront student council" cannot inexplicably lead to node4 "all clues point to the library". This breaks immersion.
   - **Bad example**: node2 discovers prank + node3 discovers student council corruption -> node4 "all point to the library" (logically abrupt)
   - **Good example**: node2 finds a library barcode on chemical packaging + node3 discovers hidden library borrowing records -> node4 naturally points to the library (logically coherent)
9. **Node summary must be detailed and clear**:
   - Each node's summary must contain enough information for the writer to understand what happened, why it happened
   - If a mystery, secret, or key information is involved, it must be explicitly stated in the summary (no vagueness)
   - The summary should answer: who, did what, why, with what result, and include rich plot details rather than a single brief sentence
10. **Make full use of merge mechanics**: avoid generating duplicate content; instead, let different choices lead to different discoveries/experiences that ultimately form a coherent narrative
11. Ensure all characters have reasonable appearances across different branches"""

    # Model parameters
    TEMPERATURE = 0.7
    MAX_TOKENS = 20000


# ==================== Producer Agent Configuration ====================
class ProducerConfig:
    """Producer Agent - responsible for reviewing the game design document and controlling project direction"""
    
    SYSTEM_PROMPT = """You are a senior Visual Novel game producer, responsible for maintaining overall game quality and project direction.
Your task is to review the design proposals submitted by the designer, ensuring they meet user requirements and have both commercial value and artistic coherence."""

    GAME_DESIGN_CRITIQUE_PROMPT = """You are the game producer. Please review the following game design document drafted by the designer.

[User's Original Requirements]
{user_requirements}

[Core Parameter Targets]
- Expected total nodes: {expected_nodes}
- Expected character count: {expected_characters}

[Game Design Document]
{game_design}

Please provide feedback across the following dimensions:
1. **Metric compliance**: Does the total node count meet requirements (variance of +-2 is acceptable, but must not deviate too much)? Does the character count meet requirements?
2. **Requirements alignment**: Does the proposal fully implement all of the user's core requirements?
3. **Graph completeness**: Is the DAG structure reasonable? Are there any dead nodes or isolated nodes?
4. **Story logic**: Does it tell a clear and complete story? Is the story sufficiently detailed?

Please give your verdict:
- If the metrics are acceptable and the proposal can go directly into development, reply only with "PASS".
- If metrics are not met or improvements are needed, provide specific, professional, and pointed detailed revision suggestions."""



# ==================== Artist Agent Configuration ====================
class ArtistConfig:
    """Artist Agent - responsible for generating character sprites"""
    
    # Standard expression list
    STANDARD_EXPRESSIONS = os.getenv("GAME_CHARACTER_EXPRESSIONS", "neutral").split(",")
    
    # Character sprite prompt template
    IMAGE_PROMPT_TEMPLATE = """A single anime character portrait in vertical orientation for a visual novel game.

Story Context: {story_background}
Art Style: {art_style}

Character Appearance: {appearance}
Character Personality: {personality}
Expression: {expression}

CRITICAL REQUIREMENTS:
- ONLY ONE character, solo portrait
- VERTICAL portrait orientation (not horizontal)
- Upper body view (from waist up) ONLY
- NO knee-up view, NO full body view
- Character facing forward, looking at viewer
- Standing pose
- SOLID WHITE BACKGROUND. This is MANDATORY.
- The background must be pure white (#FFFFFF) to facilitate background removal.
- NO complex backgrounds, NO scenery, NO other characters

POSE INSTRUCTIONS:
- Do NOT just change the facial expression.
- Generate a DYNAMIC POSE and HAND GESTURES that reflect both the '{expression}' and the character's '{personality}'.
- For example, a shy character might look away or fidget; an energetic character might wave or pump a fist.
- The body language must be expressive and natural.

Art style: High quality Japanese anime/manga style, beautiful detailed eyes, detailed hair, soft lighting, clean professional composition.

This is a character sprite for a visual novel game."""
    
    # Image generation parameters (character sprites)
    IMAGE_SIZE = "1024x1792"  # Vertical, suitable for sprites
    IMAGE_WIDTH = 1024
    IMAGE_HEIGHT = 1792
    IMAGE_QUALITY = "standard"  # "standard" or "hd"
    IMAGE_STYLE = "vivid"  # "vivid" or "natural"
    
    # Scene background image configuration
    BACKGROUND_PROMPT_TEMPLATE = """masterpiece, wallpaper, 8k, detailed CG, {location}, {atmosphere}, {time_of_day}, (no_human)

Story Context: {story_background}
Art Style: {art_style}

Beautiful anime style background scene for visual novel. High quality Japanese anime background art, detailed scenery, atmospheric lighting, rich colors, depth and dimension.

Wide establishing shot, environment only, professional game CG quality.

AVOID: people, characters, text, watermark, low quality, cropped, blurry, bad composition"""

    BACKGROUND_SIZE = "1792x1024"  # Horizontal, suitable for backgrounds
    BACKGROUND_WIDTH = 1792
    BACKGROUND_HEIGHT = 1024
    BACKGROUND_QUALITY = "standard"
    BACKGROUND_STYLE = "vivid"

    # Title screen prompt template
    TITLE_IMAGE_PROMPT_TEMPLATE = """A masterpiece, high-quality title screen illustration for a visual novel game.

Game Title: {title}
Theme/Setting: {background}

Art style: High quality Japanese anime/manga style, beautiful detailed art, atmospheric lighting, rich colors, professional game CG quality.
The image should be eye-catching and represent the mood of the game.
It should look like a professional game cover or title screen background.
Wide aspect ratio (16:9).

AVOID: text, watermark, low quality, cropped, blurry, bad composition"""


# ==================== Writer Agent Configuration ====================
class WriterConfig:
    """Writer Agent - responsible for generating story nodes"""
    
    SYSTEM_PROMPT = """You are an experienced Visual Novel writer, skilled at crafting nuanced dialogue and engaging plots.
Your task is to generate the detailed story script for the current plot node based on the game design document and the current node's outline.

Story requirements:
1. Dialogue must be natural and fluid, consistent with character personalities
2. Plot content must match the current node's summary description
3. **If the current node has child nodes (branches), you must provide choices at the end of the script; the number of choices must match the number of child nodes, one-to-one**
4. Annotate the sprite needed for each dialogue, in the format <image id="character_name">expression</image>.
   - **Prefer expressions from the [Available Sprites List].**
   - **Within the same node, if a character's emotion has not changed significantly, reuse the same expression tag; do not frequently switch or create subtly different expressions.**
   - **If the plot requires a new expression, you may freely create a new expression tag (e.g., <image id="character_name">despair</image>); it will be automatically generated.**
   - **Expression names must be complete English words; single-letter abbreviations are strictly forbidden.**
5. **Important: Scenes can only use pre-defined scenes from the game design; do not create new scenes**
6. **Output the script content directly, without any thought process or explanatory text**"""

    PLOT_SPLIT_PROMPT = """You are a professional writer. Please split the following plot node outline into {segment_count} specific "plot segments" (Plot Points).
{split_instruction}

[Node Outline]
{node_summary}

[Previous Story Summary]
{previous_story_summary}

[Available Character Details]
{available_characters}

[Available Scene List]
{available_scenes}

Please output a JSON format list:
[
  {{
    "id": 1,
    "summary": "Detailed description of segment 1...",
    "characters": ["Character Name A", "Character Name B"] (must be chosen from the [Available Character List]),
    "location": "Scene name" (must be chosen from the [Available Scene List])
  }},
  ...
]

Note: Character names in each segment's characters array must exactly match those in the [Available Character List]."""

    PLOT_SYNTHESIS_PROMPT = """You are a professional Visual Novel writer. Your task is to integrate the following plot segments performed by AI actors (in JSON format logs) into a literary, immersive Visual Novel script.

[Plot Segment Performance Logs (JSON)]
{plot_performances}

[Story Context]
{story_context}

[Subsequent Branch Options]
{choices}

[Available Character Details]
{available_characters}

[Available Scene List]
{available_scenes}

[Core Tasks]
1. **Script integration and polish**:
   - Connect the scattered dialogue logs into a coherent story.
   - **Optimize dialogue pacing**: if an actor's lines are too long or unnatural, trim and polish them to be more colloquial and in character.
   - **Enhance narrative descriptions**: do not merely list dialogue. Add rich **environmental, action, expression, and psychological descriptions** between dialogues as appropriate. Action and inner thoughts within actor dialogue must be separated as narration, not embedded in dialogue.
   - **First-person perspective**: the script is typically told from the protagonist's ("I") perspective. Convert action descriptions from actor logs into "I"'s observations or inner monologue.
   - **Optimize overall logic**: ensure the overall plot logic is complete, coherent and reasonable; add additional context where needed so the audience can immerse in the story and understand its development.

2. **Format rules**:
   - **Scene tags**: **whenever a new story node begins or the scene changes, a scene tag must appear at the very beginning of the script.**
     **Scene names must strictly use names from the [Available Scene List]; do not create new ones.**
     Format: `<scene>Scene Name</scene>`
     Example: `<scene>Deep Sea Station Main Control Room</scene>`
     **Scene tags must be on their own line, with a blank line before the story content starts.**
   - **Narration**: used for environmental, action, and psychological descriptions.
     Format: `<content id="narration">Content...</content>`
   - **Dialogue**:
     <image id="character_name">expression</image>
     <content id="character_name">Dialogue content</content>
     **Character names must strictly use names from the [Available Character List] (except the protagonist, who is always referred to as "I").**
   - **Protagonist identifier**: if the character is the protagonist, always use "I" as the name (e.g., `<content id="I">...</content>`).

3. **Branch options and endings**:
   - If [Subsequent Branch Options] are provided, generate options at the very end of the script.
   - **Option text must be concise and immersive**: summarize the option content into a phrase of 10 characters or fewer (e.g., "Go to the beach", "Stay in class").
   - **No immersion-breaking markers**: do not add [] or other tag hints before options (such as [Go with the flow], [Activate hypermemory], etc.); write the option content directly.
   - **Ending description**: if it is a leaf node (ending), use narration to naturally describe the ending; do not use immersion-breaking markers like **[BAD END]** or **[GOOD END]**.
   - Format must be: `<choice target="node_id">Option text</choice>`

[Example Output Style]
<scene>High School Classroom</scene>

<content id="narration">Afternoon sunlight streams through the gaps in the curtains onto the desks. The air smells faintly of chalk dust.</content>

<image id="Xia Yu">bored</image>
<content id="I">...So boring.</content>

<content id="narration">I slump over my desk, absentmindedly spinning a ballpoint pen. Just then, the classroom door is shoved open.</content>

<image id="Ren">excited</image>
<content id="Ren">Everyone! Come look at this!</content>

Output the final script directly, without any explanatory text."""

    NEXT_SPEAKER_PROMPT = """You are the director overseeing the entire scene.
[Current Plot Segment Goal]
{plot_summary}

[Available Character Details]
{characters}

[Story Context]
{story_context}

**Your core responsibilities:**
1. Judge whether the [Plot Segment Goal] has been achieved
2. If the goal is complete, **call cut immediately** — do not drag it out

**If the plot segment goal is substantially complete, call cut immediately:**
<character>STOP</character>

**Otherwise, if key plot is still unfinished, designate the next speaker:**
<character>Character Name</character>
<advice>The plot content this character needs to advance — keep it very brief, ideally just one action's intent; leave space for others to speak; strictly do not dictate specific lines</advice>

**Examples:**
<character>Xia Yu</character>
<advice>Express their final decision on the proposal</advice>

Output only the XML tags; no other content.
Note: you can only choose one of the on-scene characters as the next speaker; you cannot choose narration or off-scene characters.
Maintain overall plot integrity and coherence."""

    SUMMARY_PROMPT = """Please generate a brief summary for the following story content, to be used as the "previously on" for subsequent plot.

[Story Content]
{story_content}

Requirements:
1. Summarize the main events and key dialogue.
2. Include any important foreshadowing or state changes.
3. Keep the length within 200 words.
4. Output the summary content directly."""


# ==================== Actor Agent Configuration ====================
class ActorConfig:
    """Actor Agent - responsible for playing specific characters and reviewing scripts"""
    
    # Model parameters
    TEMPERATURE = 0.7
    
    SYSTEM_PROMPT = """You are now the character "{name}" in a Visual Novel game.

[Your Character Profile]
Personality: {personality}
Background: {background}

Immerse yourself in the character, think and act in the first person, but avoid being stereotypical or exaggerating personality traits — just ensure the character does not go OOC.
Forget that you are an AI model; you are this character."""

    PERFORM_PROMPT = """Based on the following plot segment outline and the current dialogue log, continue performing your lines and actions in the scene.

[Plot Segment]
{plot_summary}

[Other On-Scene Character Details]
{other_characters}

[Available Expressions (Character Expressions)]
{character_expressions}

[Story Context]
{story_context}

Begin performing directly; do not output any analysis, OOC checks, or additional explanatory text.
Output your performance in script format (including dialogue and action descriptions):
<image id="{name}">expression</image>
<content id="{script_label}">Dialogue content</content>

Notes:
1. Only output your own part; do not repeat previous dialogue.
2. **Minimal dialogue principle**: say only one sentence or do one action at a time. No lengthy speeches. Do not express everything at once; leave space for the other party to respond.
3. Stay consistent with your personality, while keeping dialogue natural and fluid — avoid being overly formal, dramatic, or stereotypical.
4. Use <image id="{name}">expression</image> to mark expression changes — **must be on its own line**.
   - Prefer reusing expressions from the [Available Sprites List].
   - If your emotion has not changed significantly from existing sprites, reuse the existing one.
   - **If none of the existing sprites can represent your current emotion, you need a new sprite — create a new expression name to better express yourself.**
   - **Expression names must be complete English words (e.g., 'angry', 'surprised'); single-letter abbreviations (e.g., 't', 'a') or Chinese are strictly forbidden.**"""

    IMAGE_CRITIQUE_PROMPT = """You are now reviewing the character sprite generated for you. Please evaluate the image from a first-person, in-character perspective.

[Story Background]
{story_background}

[Art Style]
{art_style}

[Your Appearance Setting]
{appearance}

[Current Required Expression]
{expression}

Please review this sprite in your character's voice and perspective:
1. **Stay in character**: use "I" to refer to yourself, speaking in your personality and tone.
2. **Review criteria**:
   - Is the image itself logically consistent? Are there multiple hands/feet, low quality, or other image issues?
   - Is the sprite itself visually appealing and consistent with visual novel sprite style?
   - Does the art style match the story setting ({art_style})?
   - Does the appearance match your profile (hair color, eyes, clothing, build, etc.)?
   - Does the expression accurately convey the emotion {expression}?
   - Does the overall style and vibe match your character profile?

If completely satisfied, say: **"PASS"** (must include this word)

If not satisfied, point out the issues in your character's voice, for example:
- "The expression is too stiff; when I'm feeling {expression}, I wouldn't look like this..."

Begin the review directly, speaking in your personality and tone."""

    EXPRESSION_DESCRIPTION_PROMPT = """You are playing {name}.
Your task is to describe your specific appearance when displaying the [{expression}] expression.
Please provide a detailed visual description, including facial feature details, face demeanor, eye expression, mouth shape, and possible body movements.
The description will be used to generate the sprite image.

Character profile:
{character_info}

Output the description text directly, without any other content."""

# ==================== File Path Configuration ====================
class PathConfig:
    """File path configuration"""
    
    # Project root directory
    import sys
    if getattr(sys, 'frozen', False):
        PROJECT_ROOT = sys._MEIPASS
    else:
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Data directories
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    IMAGES_DIR = os.path.join(DATA_DIR, "images")
    CHARACTERS_DIR = os.path.join(IMAGES_DIR, "characters")
    BACKGROUNDS_DIR = os.path.join(IMAGES_DIR, "backgrounds")  # Background image directory
    
    # Game data files
    GAME_DESIGN_FILE = os.path.join(DATA_DIR, "game_design.json")
    STORY_FILE = os.path.join(DATA_DIR, "story.txt")
    CHARACTER_INFO_FILE = os.path.join(DATA_DIR, "character_info.json")
    
    # Log directories
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
    TEXT_LOG_DIR  = os.path.join(LOG_DIR, "text_log")    # Stores performance_node.jsonl
    IMAGE_LOG_DIR = os.path.join(LOG_DIR, "image_log")   # Stores rejected images
    QUALITY_LOG_DIR = os.path.join(LOG_DIR, "quality")   # Stores quality scoring logs

    # Quality log files
    QUALITY_ROUNDS_LOG = os.path.join(QUALITY_LOG_DIR, "rounds.jsonl")   # Per-round scores
    QUALITY_FINAL_LOG  = os.path.join(QUALITY_LOG_DIR, "final.jsonl")    # Per-run final scores
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all necessary directories exist"""
        dirs = [
            cls.DATA_DIR,
            cls.IMAGES_DIR,
            cls.CHARACTERS_DIR,
            cls.BACKGROUNDS_DIR,
            cls.LOG_DIR,
            cls.TEXT_LOG_DIR,
            cls.IMAGE_LOG_DIR,
            cls.QUALITY_LOG_DIR,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)


# Initialize necessary directories on startup
PathConfig.ensure_directories()


# ==================== RAG Configuration ====================
class RAGConfig:
    """RAG (Retrieval-Augmented Generation) configuration — for fan-fiction game creation mode"""

    # Wikipedia default language: "en" (English) or "zh" (Chinese)
    # Note: Most IPs' English Wikipedia pages are more detailed than Chinese
    WIKIPEDIA_LANGUAGE = os.getenv("RAG_WIKIPEDIA_LANGUAGE", "en")

    # Number of document chunks returned per retrieval
    N_RESULTS_CHARACTER = int(os.getenv("RAG_N_RESULTS_CHARACTER", "5"))
    N_RESULTS_WORLD = int(os.getenv("RAG_N_RESULTS_WORLD", "5"))
    N_RESULTS_OVERVIEW = int(os.getenv("RAG_N_RESULTS_OVERVIEW", "3"))

    # Whether to force rebuild the knowledge base (re-fetch Wikipedia on every run)
    # Default False: if a local knowledge base already exists, reuse it to save API calls
    FORCE_REBUILD = os.getenv("RAG_FORCE_REBUILD", "false").lower() == "true"

    # Knowledge base storage directory (under DATA_DIR/rag/)
    RAG_DIR = os.path.join(PathConfig.DATA_DIR, "rag")

    @classmethod
    def ensure_rag_dir(cls):
        os.makedirs(cls.RAG_DIR, exist_ok=True)
