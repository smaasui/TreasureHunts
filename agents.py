import asyncio
from agents import Agent, function_tool
from connection import client

@function_tool
def extract_product_info(user_input: str):
    """
    Extract structured shopping info from user's natural text.
    Supports only:
      - Shoes: name, size, color, quantity
      - Other: name, size_or_weight, color, quantity
    Missing info should be null.
    """

    system_prompt = """
    You are a multilingual shopping assistant.
    Your job is to convert the user's shopping request into structured JSON.
    Translate colors and names to English when possible.

    Categories:
    - "shoes"
    - "other"

    JSON format:
    {
      "products": [
        {
          "category": "shoes" or "other",
          "name": "<item name>",
          "size": "<number or null>",
          "color": "<color in English or null>",
          "quantity": "<integer or null>",
          "size_or_weight": "<value or null if not applicable>"
        }
      ]
    }

    Always output valid JSON only.
    """

    async def run_llm():
        response = await client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    return asyncio.get_event_loop().run_until_complete(run_llm())


def create_shopping_agent():
    """Return a ready-to-use agent for shopping understanding."""
    return Agent(
        name="Shopping Agent",
        instructions=(
            "Understand user's shopping intent and extract structured info "
            "in JSON format using the extract_product_info tool."
        ),
        tools=[extract_product_info],
    )
