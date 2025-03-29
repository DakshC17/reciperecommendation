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
        ],
        "response_format": {"type": "json_object"}  # This ensures JSON output
    }

    print("📤 Sending request to Groq API:", json.dumps(payload, indent=2))

    async with httpx.AsyncClient(timeout=15) as client:
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
                content = data["choices"][0]["message"]["content"]
                try:
                    # Try to parse immediately to catch JSON issues early
                    json.loads(content)
                    return content
                except json.JSONDecodeError as e:
                    raise HTTPException(status_code=500, detail=f"Groq API returned invalid JSON: {str(e)}")
            else:
                raise HTTPException(status_code=500, detail="Empty response from Groq API.")

        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Groq API Error: {e.response.text}")
        except httpx.TimeoutException:
            raise HTTPException(status_code=500, detail="Groq API request timed out.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected Error: {str(e)}")

async def extract_json(response: str):
    """Extracts and validates JSON from a response."""
    try:
        # First try to parse the entire response as JSON
        return json.loads(response)
    except json.JSONDecodeError:
        # If that fails, try to extract JSON from the response
        try:
            # Improved regex pattern to handle nested structures
            json_pattern = r'```json\n({.*?})\n```|```\n({.*?})\n```|({.*})'
            matches = re.findall(json_pattern, response, re.DOTALL)
            
            # Flatten matches and filter out empty groups
            possible_jsons = [m for group in matches for m in group if m]
            
            if possible_jsons:
                # Try each possible JSON match until one works
                for possible_json in possible_jsons:
                    try:
                        return json.loads(possible_json)
                    except json.JSONDecodeError:
                        continue
            
            # If no match found in code blocks, try to find the outermost JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            
            raise HTTPException(status_code=500, detail="No valid JSON found in response.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to extract JSON: {str(e)}")

async def filter_food_items(items: list[str]):
    """Filters food-related items using Groq's API and extracts valid JSON."""
    system_prompt = """You are an expert in classifying grocery items. 
    Always respond with valid JSON only, no additional text or explanations.
    The JSON must follow this exact structure:
    {
        "food_items": ["item1", "item2"],
        "non_food_items": ["item3", "item4"]
    }"""
    
    user_prompt = f"""Classify these items into food-related and non-food items: {json.dumps(items)}.
    Rules:
    - Food-related: Edible ingredients, cooking essentials
    - Non-food: Household items, medicines, toiletries
    Respond ONLY with the JSON output, no additional text or explanations."""

    response = await groq_chat(system_prompt, user_prompt)
    return await extract_json(response)

async def get_recipes(food_items: list[str]):
    """Fetches recipes using filtered food items."""
    if not food_items:
        return {"recipes": [], "additional_ingredients": []}

    system_prompt = """You are an expert chef providing detailed recipes. 
    Always respond with valid JSON only, no additional text or explanations.
    The JSON must follow this exact structure:
    {
        "recipes": [
            {
                "name": "Recipe Name",
                "ingredients": [
                    {"name": "ingredient1", "quantity": "1 cup"},
                    {"name": "ingredient2", "quantity": "2 tbsp"}
                ],
                "instructions": ["Step 1", "Step 2"],
                "servings": 2,
                "prep_time": "10 mins",
                "cook_time": "20 mins",
                "missing_ingredients": ["ingredient3"]
            }
        ],
        "additional_ingredients": ["ingredient3", "ingredient4"]
    }"""

    user_prompt = f"""Generate 3 detailed recipes using these ingredients: {json.dumps(food_items)}.
    Each recipe must include:
    1. Recipe Name
    2. Ingredients List with precise quantities
    3. Step-by-Step Cooking Instructions
    4. Missing Ingredients
    5. Serving Size & Time Estimates
    Respond ONLY with the JSON output, no additional text or explanations."""

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