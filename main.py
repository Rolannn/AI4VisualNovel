import sys
import os
import logging
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from workflow import WorkflowController
from agents.config import ProducerConfig, DesignerConfig, PathConfig


def setup_logging(level=logging.INFO):
    """Configure the logging system."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            logging.FileHandler(os.path.join(PathConfig.LOG_DIR, 'ai_visual_novel.log'), encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

# The entry point for the entire visual novel generation pipeline.
# args => initialize system(workflow + agents) => Read User Input (requirements / fan-fiction)
# => Invoke Core Generation Logic (workflow.create_new_game) => Print Results + Output File Path
def create_game_flow(args):
    """Run the game creation flow."""
    print("\n" + "="*70)
    print("AI Visual Novel - Game Creation Mode")
    print("="*70)

    workflow = WorkflowController()

    # Initialize agents
    workflow.initialize_agents(
        openai_api_key=args.openai_key,
        openai_base_url=args.openai_base_url
    )

    # Load user requirements from file (if provided)
    user_requirements = ""
    if args.requirements_file:
        if os.path.exists(args.requirements_file):
            print(f"\nLoading game concept from file: {args.requirements_file}")
            try:
                with open(args.requirements_file, 'r', encoding='utf-8') as f:
                    user_requirements = f.read().strip()
                    print(f"Loaded: {user_requirements[:50]}...")
            except Exception as e:
                print(f"Warning: Failed to read file: {e}. Continuing with empty requirements.")
        else:
            print(f"Warning: Requirements file not found: {args.requirements_file}. Continuing with empty requirements.")
    else:
        print("\nNo requirements file provided. AI will generate content freely.")

    # Parse fan-fiction character list (comma-separated)
    fan_characters = None
    if args.fan_characters:
        fan_characters = [c.strip() for c in args.fan_characters.split(",") if c.strip()]

    # Fan-fiction mode notice
    if args.franchise:
        print(f"\nFan-fiction mode enabled")
        print(f"   IP / Franchise: {args.franchise}")
        print(f"   Characters: {fan_characters or '(let AI decide)'}")
        if args.fan_docs:
            print(f"   Local docs: {args.fan_docs}")

    use_quality_scorer = not args.no_quality_check
    if not use_quality_scorer:
        print("\n[Baseline mode] Quality scorer intervention DISABLED"
              " — final scores still logged for comparison.")

    if args.force_regen or args.regen_story_only:
        import shutil
        from agents.config import PathConfig as _PC
        # story-only: only clear story.txt; force-regen: also clear game_design
        targets = [_PC.STORY_FILE]
        if args.force_regen:
            targets.append(_PC.GAME_DESIGN_FILE)
        for p in targets:
            if os.path.exists(p):
                os.remove(p)
                print(f"  [regen] Deleted: {p}")
        if args.force_regen:
            perf_dir = os.path.join(os.path.dirname(_PC.DATA_DIR), "logs", "performance")
            if os.path.isdir(perf_dir):
                shutil.rmtree(perf_dir)
                print(f"  [regen] Cleared: {perf_dir}")

    # Create the game
    game_design = workflow.create_new_game(
        character_count=args.character_count,
        requirements=user_requirements,
        franchise=args.franchise or "",
        fan_characters=fan_characters,
        fan_docs_dir=args.fan_docs or "",
        rag_force_rebuild=args.rag_rebuild,
        rag_language=args.rag_language or "",
        use_quality_scorer=use_quality_scorer,
    )

    print("\n" + "="*70)
    print("Game creation complete!")
    print("="*70)
    print(f"\nTitle: {game_design['title']}")
    print(f"Background:\n{game_design['background'][:200]}...")
    print(f"\nCharacters:")
    for char in game_design['characters']:
        print(f"   - {char['name']}: {char['personality']}")

    print(f"\nGame data saved to: {PathConfig.DATA_DIR}")
    print(f"Character sprites saved to: {PathConfig.CHARACTERS_DIR}")
    print(f"\nTip: Run 'python main.py --mode play' to start playing.")


def play_game_flow():
    """Run the game playback flow."""
    print("\n" + "="*70)
    print("AI Visual Novel - Play Mode")
    print("="*70)

    # Check if game data exists
    if not os.path.exists(PathConfig.GAME_DESIGN_FILE):
        print("\nError: No game data found!")
        print("   Please run: python main.py --mode create")
        return

    # Launch the game UI
    print("\nLaunching game...")

    from game_engine.manager import GameManager
    game = GameManager()
    game.run()


def status_flow():
    """Display the current game status."""
    print("\n" + "="*70)
    print("AI Visual Novel - Game Status")
    print("="*70)

    workflow = WorkflowController()

    if not workflow.load_existing_game():
        print("\nError: No game data found!")
        return

    status = workflow.get_game_status()

    print(f"\nTitle: {status['title']}")
    print(f"Progress: {status['completed_nodes']}/{status['total_nodes']} story nodes completed")

    if status['completed_nodes'] == status['total_nodes']:
        print(f"\nAll story nodes have been generated!")
    else:
        print(f"\nSome nodes are not yet generated.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='AI Visual Novel - An AI-driven automated Visual Novel generation and playback system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a new original game
  python main.py --mode create --requirements-file data/story.txt

  # Fan-fiction mode - create a game based on Genshin Impact characters
  python main.py --mode create --franchise "Genshin Impact" --fan-characters "Hu Tao,Zhongli,Keqing"

  # Fan-fiction mode - Harry Potter with local supplementary documents
  python main.py --mode create --franchise "Harry Potter" --fan-characters "Hermione Granger,Draco Malfoy" --fan-docs data/hp_docs/

  # Fan-fiction mode - force rebuild the RAG knowledge base
  python main.py --mode create --franchise "Genshin Impact" --fan-characters "Hu Tao" --rag-rebuild

  # Play the generated game
  python main.py --mode play

  # Check game generation status
  python main.py --mode status

  # A/B comparison: run twice and compare logs in eval/quality_logs/
  python main.py --mode create --no-quality-check   # baseline (control)
  python main.py --mode create                      # with quality scorer (treatment)

Environment variables:
  OPENAI_API_KEY          OpenAI API key (for GPT and image generation)
  OPENAI_BASE_URL         OpenAI API base URL (optional)
  RAG_WIKIPEDIA_LANGUAGE  Wikipedia language for RAG (default: en)
  RAG_FORCE_REBUILD       Force rebuild RAG knowledge base on every run (default: false)
        """
    )

    parser.add_argument(
        '--mode',
        choices=['create', 'play', 'status'],
        default='play',
        help='Run mode: create=generate game, play=launch game, status=show progress'
    )

    parser.add_argument('--character-count', type=int, default=DesignerConfig.DEFAULT_CHARACTER_COUNT, help='Number of characters (including protagonist)')
    parser.add_argument('--requirements-file', help='Path to a text file containing the game concept / requirements')

    parser.add_argument('--openai-key', help='OpenAI API key (overrides environment variable)')
    parser.add_argument('--openai-base-url', help='OpenAI API base URL (overrides environment variable)')

    # -- Fan-fiction mode (RAG + Wikipedia) --
    fan_group = parser.add_argument_group('Fan-Fiction Mode')
    fan_group.add_argument(
        '--franchise',
        help='Name of the source IP / franchise (e.g. "Genshin Impact", "Harry Potter"). Enables fan-fiction mode.',
        default=''
    )
    fan_group.add_argument(
        '--fan-characters',
        help='Comma-separated list of canon character names to include (e.g. "Hu Tao,Zhongli")',
        default=''
    )
    fan_group.add_argument(
        '--fan-docs',
        help='Directory of local supplementary documents (.txt/.md/.json) to add to the RAG knowledge base',
        default=''
    )
    fan_group.add_argument(
        '--rag-rebuild',
        action='store_true',
        help='Force rebuild the RAG knowledge base (clears cache and re-fetches Wikipedia)'
    )
    fan_group.add_argument(
        '--rag-language',
        choices=['en', 'zh'],
        default='',
        help='Wikipedia language for RAG (default: en)'
    )
    # -----------------------------------------

    parser.add_argument(
        '--no-quality-check',
        action='store_true',
        help='Disable quality scorer intervention (baseline / control run for A/B comparison).'
             ' Final scores are still computed for fair comparison.'
    )
    parser.add_argument(
        '--force-regen',
        action='store_true',
        help='Delete existing story.txt and game_design.json before generating, '
             'forcing a completely fresh run (needed for A/B comparison).'
    )
    parser.add_argument(
        '--regen-story-only',
        action='store_true',
        help='Delete only story.txt (keep game_design.json). '
             'Use this to A/B test scorer on the SAME game design.'
    )
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)

    # Dispatch to the appropriate flow
    try:
        if args.mode == 'create':
            create_game_flow(args)
        elif args.mode == 'play':
            play_game_flow()
        elif args.mode == 'status':
            status_flow()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting.")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
