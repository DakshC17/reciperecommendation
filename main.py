import asyncio
import json
import httpx
import os
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load API Key from Environment Variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise HTTPException(status_code=500, detail="❌ GROQ_API_KEY not set. Export it before running.")

app = FastAPI()

class GroceryList(BaseModel):
    items: list[str]

async def groq_chat(system_prompt: str, user_prompt: str):
    """Sends a prompt to Groq's LLaMA 3 API and returns the response."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    print("📤 Sending request to Groq API:", json.dumps(payload, indent=2))

    async with httpx.AsyncClient(timeout=15) as client:  # Added timeout
        try:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            data = response.json()

            print("📥 Groq API Response:", json.dumps(data, indent=2))

            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"]
            else:
                raise HTTPException(status_code=500, detail="Empty response from Groq API.")

        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Groq API Error: {e.response.text}")
        except httpx.TimeoutException:
            raise HTTPException(status_code=500, detail="Groq API request timed out.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected Error: {str(e)}")


async def extract_json(response: str):
    """Extracts valid JSON from a response using regex."""
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        json_response = json_match.group(0)
        try:
            return json.loads(json_response)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Groq API returned malformed JSON.")
    raise HTTPException(status_code=500, detail="Could not extract JSON from Groq API response.")


async def filter_food_items(items: list[str]):
    """Filters food-related items using Groq's API and extracts valid JSON."""
    system_prompt = "You are an expert in classifying grocery items."
    user_prompt = f"""
    Classify these grocery items into **food-related** and **non-food items**: {', '.join(items)}.

    **Rules:**
    - ✅ **Food-related:** Edible ingredients, cooking essentials.
    - ❌ **Non-food:** Household items, medicines, toiletries.

    **Output JSON Format (No extra text!):**
    {{
        "food_items": ["food1", "food2"],
        "non_food_items": ["nonfood1", "nonfood2"]
    }}
    """
    
    response = await groq_chat(system_prompt, user_prompt)
    return await extract_json(response)


async def get_recipes(food_items: list[str]):
    """Fetches recipes using filtered food items."""
    if not food_items:
        return {"recipes": [], "additional_ingredients": []}  # Return empty list if no food items

    system_prompt = "You are an expert chef providing detailed recipes."
    user_prompt = f"""
    Generate **3 recipes** using these ingredients: {', '.join(food_items)}.

    **Each recipe must have:**
    - Name
    - Ingredients with quantity
    - Cooking instructions
    - Missing ingredients

    **Output JSON Format:**
    {{
      "recipes": [
        {{
          "name": "Recipe 1",
          "ingredients": [
            {{"name": "ingredient_1", "quantity": "X unit"}},
            {{"name": "ingredient_2", "quantity": "Y unit"}}
          ],
          "instructions": "Step 1: ... Step 2: ...",
          "missing_ingredients": ["ingredient_x"]
        }},
        ...
      ],
      "additional_ingredients": ["ingredient_x", "ingredient_y"]
    }}
    """
    
    response = await groq_chat(system_prompt, user_prompt)
    return await extract_json(response)


@app.post("/suggest-recipes")
async def suggest_recipes(grocery_list: GroceryList):
    """Filters food items and fetches recipes in parallel."""
    try:
        # First, filter food items
        food_items_result = await filter_food_items(grocery_list.items)
        food_items = food_items_result.get("food_items", [])
        non_food_items = food_items_result.get("non_food_items", [])

        if not food_items:
            raise HTTPException(status_code=400, detail="No food-related items found.")

        # Fetch recipes using filtered food items
        recipes_result = await get_recipes(food_items)

        return {
            "filtered_food_items": food_items,
            "non_food_items": non_food_items,
            "recipes": recipes_result.get("recipes", []),
            "additional_ingredients": recipes_result.get("additional_ingredients", [])
        }

    except HTTPException as e:
        raise e  # Re-raise known HTTP errors
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
