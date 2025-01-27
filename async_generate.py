import asyncio
import json
import logging
from pathlib import Path
from typing import Literal, Optional

from ollama import AsyncClient
from pydantic import BaseModel, Field, model_validator
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
NUM_EMAILS = 30  # Generate 30 emails
MODEL_NAME = "deepseek-r1:8b"  # Replace with your preferred model


class EmailSchema(BaseModel):
    """Pydantic model for validating email structure."""
    date: str = Field(description="ISO 8601 timestamp of email creation")
    subject: str = Field(description="Email subject line")
    sender: str = Field(alias="from", description="Sender's email address")
    body: str = Field(description="Main content of the email")

    class Config:
        populate_by_name = True  # Allow 'from' alias usage
        frozen = True  # Make instances immutable


class GenerationConfig(BaseModel):
    """Configuration for model generation parameters"""
    temperature: float = 1.0
    max_tokens: int = 2048
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=lambda _: logger.warning("Retrying due to failure...")
)
async def generate_email(client: AsyncClient, index: int) -> Optional[EmailSchema]:
    """Generate and validate a single email using AI model."""
    try:
        response = await client.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Michael James, a highly sought-after Wealth Manager receiving numerous daily emails. "
                        "Generate realistic business emails including: contact requests, quotes, information inquiries, "
                        "collaboration proposals, product purchases, and other professional correspondence. "
                        "Include realistic details: client names, industries, projects, figures, and contracts. "
                        "Format response as JSON with fields: date (ISO 8601), subject, from, body. "
                        "Example: {\"date\": \"2024-01-15T14:30:00Z\", \"subject\": \"Portfolio Review Request\", "
                        "\"from\": \"client@example.com\", \"body\": \"Dear Michael, I would like to schedule...\"}"
                    )
                },
                {"role": "user", "content": "Generate a professional email"}
            ],
            format=EmailSchema.model_json_schema(),
            options=GenerationConfig().model_dump()
        )

        email_data = EmailSchema.model_validate_json(response.message['content'])
        save_email(email_data, index)
        return email_data

    except Exception as e:
        logger.error(f"Failed to generate email {index}: {str(e)}")
        return None


def save_email(data: EmailSchema, index: int) -> None:
    """Save validated email data to JSON file."""
    try:
        DATA_DIR.mkdir(exist_ok=True)
        file_path = DATA_DIR / f"email_{index:03d}.json"
        
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data.model_dump(by_alias=True), f, ensure_ascii=False, indent=2)
            
        logger.info(f"Successfully saved email {index}")

    except IOError as e:
        logger.error(f"File save error for email {index}: {str(e)}")


async def main():
    """Main async function to generate multiple emails."""
    client = AsyncClient()
    tasks = [generate_email(client, i) for i in range(NUM_EMAILS)]
    results = await asyncio.gather(*tasks)
    
    success_count = sum(1 for result in results if result is not None)
    logger.info(f"Generation complete: {success_count}/{NUM_EMAILS} emails created")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")