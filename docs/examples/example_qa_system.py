import asyncio
from pathlib import Path
import logging
from typing import Dict, Any

from ascent_ai.config import Settings
settings = Settings()

from ascent_ai.models.inference.qa import QuestionAnsweringSystem


async def answer_question(config: Dict[str, Any]):
    logger = logging.getLogger(__name__)

    qa_system = await QuestionAnsweringSystem.initialize(config)

    # Test question
    # question = "What are the 20 top Comorbidities of patients one year before their first acute ischemic stroke?"
    # original_question = "How many women with atopic dermatitis and dysphagia"
    # original_question = "How many patients have hypertension?"
    # original_question = "What is the weather in Berlin?"
    # original_question = "How many patient are on aldactone, spironolactone, eplerenone, or finerenone?"
    # original_question = "How many patient had ventilation within 2,7,14,30,60 or 90 days after first stroke diagnosis? Provide total count and percentage"
    question = "How many people were diagnosis with stroke in the hospital?"

    try:
        logger.info(f"Original question: {question}")

        # Step 1: Generate query template
        logger.info("Generating query template...")
        template_result = await qa_system.generate_query_template(question)
        query_template = template_result["query_template"]
        logger.info(f"Generated template:\n{query_template}")

    except Exception as e:
        logger.error(f"Error in modular processing: {str(e)}")
        raise


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    querylib_file = base_dir / "querylib_20251019.db"
    log_directory = base_dir / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)

    example_config = {
        "querylib_file": querylib_file,
        "log_directory": log_directory,
        "assistant": {
            "type": "claude_sonnet",
            "model": "us.anthropic.claude-sonnet-4-20250514-v1:0"
        },
        "database": settings.SNOWFLAKE_DATABASE,
        "settings": settings
    }

    asyncio.run(answer_question(example_config))

