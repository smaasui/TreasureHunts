import os
import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, function_tool

# -----------------------------
# Load API key
# -----------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in .env")

# -----------------------------
# Setup Gemini-compatible client
# -----------------------------
client = AsyncOpenAI(
    api_key=GOOGLE_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

config = RunConfig(model=model, model_provider=client, tracing_disabled=True)

# -----------------------------
# Async Function Tool for LLM extraction
# -----------------------------
@function_tool
async def extract_product_info(user_input: str):
    """
    Extract structured product details from an unstructured query.
    Normalize all colors, names, and categories to English for database compatibility.
    """

    system_prompt = """
    You are a multilingual shopping assistant that extracts product information from user text.
    Always output clean, standardized English JSON for database usage.

    You must:
    - Translate any non-English (e.g. Urdu, Hindi, Roman Urdu) color or name into English.
    - Keep consistent English color names (e.g., "neela" -> "blue", "laal" -> "red").
    - Recognize numeric words like "ek", "do", "teen" etc. as 1, 2, 3.
    - Fill missing values as null.

    There are only two categories:
    - "shoes": for footwear-related items.
    - "other": for everything else.

    Output Format:

    {
      "products": [
        {
          "category": "shoes" | "other",
          "name": "<item name in English>",
          "size_or_weight": "<size, weight, or null>",
          "color": "<color in English or null>",
          "quantity": <integer or null>
        }
      ]
    }

    Only respond with valid JSON.
    """

    response = await client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# -----------------------------
# Agent definition
# -----------------------------
def create_shopping_agent():
    return Agent(
        name="Shopping Agent",
        instructions=(
            "You are a smart shopping assistant. "
            "Use the extract_product_info tool to interpret user queries about items to buy. "
            "Always output normalized English JSON for database compatibility."
        ),
        tools=[extract_product_info],
    )

# -----------------------------
# Main entry — single input
# -----------------------------
async def main():
    agent = create_shopping_agent()
    user_query = input("🛒 Enter your shopping request: ")

    result = await Runner.run(agent, input=user_query, run_config=config)
    print("\n🤖 JSON Output:")
    print(result)
    print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())
