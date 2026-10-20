import asyncio
import os
from pathlib import Path
import logging
import json

from ascent_ai.config import Settings
settings = Settings()

from ascent_ai.models.inference.rag import QueryLibraryManager
from ascent_ai.models.inference.criteria_to_counts import CriteriaProcessor


async def calculate_cohort(config: dict):
    base_dir = Path(__file__).resolve().parent.parent.parent
    main_path = os.getcwd()
    log_folder = settings.log_directory

    logger = logging.getLogger(__name__)

    logger.info(f"Base dir: {base_dir}")
    logger.info("Configuration:")
    logger.info("\n" + json.dumps(config, indent=4))

    try:
        nb_runs = 1
        for run in range(nb_runs):

            snowflake_database = settings.SNOWFLAKE_DATABASE
            querylib_file = base_dir / config["querylib_file"]

            QueryLibraryManager.get_instance().load_querylib(
                querylib_file=querylib_file
            )
            criteria_processor = CriteriaProcessor(
                text2sql_assistant_type=config["assistants"]["text2sql"]["type"],
                mask_assistant_type=config["assistants"]["mask"]["type"],
                text_to_criteria_assistant_type=config["assistants"][
                    "text_to_criteria"
                ]["type"],
                text2sql_assistant_model_name=config["assistants"]["text2sql"]["model"],
                mask_assistant_model_name=config["assistants"]["mask"]["model"],
                text_to_criteria_assistant_model_name=config["assistants"][
                    "text_to_criteria"
                ]["model"],
            )

            criteria_dict = await criteria_processor.text_to_criteria(
                config["input_text"]
            )

            logger.info(f"Criteria dict: {criteria_dict}")


            result = await criteria_processor.criteria_to_data(
                criteria_dict=criteria_dict,
                preferred_coding_system=config["preferred_coding_system"],
                rag_assistant_type=config["assistants"]["text2sql"]["type"],
                rag_model_name=config["assistants"]["text2sql"]["model"],
                snowflake_database=snowflake_database,
                main_path=main_path,
                log_folder=log_folder,
                querylib_file=querylib_file,
                max_nb_row=config["max_nb_row"],
            )

            (
                message,
                query_template_pred,
                query_filled,
                df_patients,
                df_recs_list_out,
                question_masked,
                rag_agent,
                verification_results,
                verdict,
                sql_executed,
            ) = result

            logger.info(f"Complete template query: \n{query_template_pred}")

            logger.info("Query explanation...")
            query_explanation = await criteria_processor.get_query_explanation(query_template_pred, criteria_dict)
            logger.info(f"Query explanation: {query_explanation}")

            if df_patients is not None:
                logger.info(f"df_patients: \n{df_patients[:min(10, len(df_patients))]}")
                logger.info(f"patient number: {len(df_patients)}")
            else:
                logger.info("No patient data available (template-only mode)")

            logger.info(f"Add here split query part: ")

            (
                query_template_preds,
                query_filled_preds,
                output_dfs,
                criteria_type_list,
                criteria_text_list,
            ) = await criteria_processor.split_query_by_criteria(
                preferred_coding_system=config["preferred_coding_system"],
                snowflake_database=snowflake_database,
                query_template_pred=query_template_pred,
                criteria_dict=criteria_dict,
            )

            logger.info(f"Query template split: \n{query_template_preds}")

            logger.info("Skipping funnel plot generation - no patient data available (template-only mode)")
            logger.info(f"Generated {len(query_template_preds)} template queries for funnel analysis")

            # Log the funnel SQL queries that would be used
            logger.info("Funnel analysis SQL templates:")
            for i, (query_template, criteria_type, criteria_text) in enumerate(zip(
                query_template_preds, criteria_type_list, criteria_text_list
            )):
                logger.info(f"  Step {i+1} ({criteria_type}): {criteria_text}")
                logger.info(f"  SQL Template: {query_template}...")

            logger.info(f"Done")

    except Exception as e:
        logger.error(f"Error in calculate_cohort: {str(e)}")
        raise


if __name__ == "__main__":

    default_config = {
        "querylib_file": "querylib_20251019.db",
        "max_nb_row": 10**7,
        "preferred_coding_system": {
            "condition": ["SNOMED"],
            "procedure": ["CPT4", "SNOMED"],
            "drug": ["RxNorm", "RxNorm Extension"],
        },
        "assistants": {
            "text_to_criteria": {
                "type": "claude_sonnet",
                "model": "us.anthropic.claude-sonnet-4-20250514-v1:0",
            },
            "mask": {
                "type": "claude_sonnet",
                "model": "us.anthropic.claude-sonnet-4-20250514-v1:0",
            },
            "text2sql": {"type": "claude_sonnet",
                        "model": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    },
        },
        "input_text": """Inclusion Criteria:
    - Women having at least two diagnosis of endometriosis (2010-2019)
    - Aged 16–45 years at index date 
    - Women having continuous enrolment in the database for a minimum of 12 months pre- and post-index date
     
    Exclusion Criteria:
    - Women with hysterectomy at any time prior to index date or up to 30 days post-index date
    - Women with menopause prior to index date
    - Women with any cancer diagnosis except for non-melanoma skin cancer at any time
    
    Index date:
    - First diagnosis of endometriosis
    """,
    }

    asyncio.run(calculate_cohort(default_config))
