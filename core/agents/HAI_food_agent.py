from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv

from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    SystemMessage,
    HumanMessage,
)

from langchain_openai import ChatOpenAI

from langchain_core.tools import tool

from langgraph.graph.message import add_messages

from langgraph.graph import StateGraph, END

from langgraph.prebuilt import ToolNode

from pydantic import BaseModel, Field

load_dotenv()

class AgentState(TypedDict):
    # Patients input 
    patient_foodInput_description: str
    patient_foodInput_image_exists: bool  # Base64 encoded image string
    patient_foodInput_image: str  # Base64 encoded image string

    # output 
    analysis_status: str
    color_status: str
    analysis_text: str
    food_macros: list[dict]  # List of dictionaries containing food macro information


model = ChatOpenAI(model="gpt-4o")

class FoodMacroOutput(BaseModel):
    title: str = Field(..., description="The name of the food item")
    weight: float = Field(..., description="The weight of the food item in grams")
    kilocalories_per100g: float = Field(..., description="The number of kilocalories per 100 grams of the food item")
    protein_per100g: float = Field(..., description="The amount of protein per 100 grams of the food item")
    carbohydrates_per100g: float = Field(..., description="The amount of carbohydrates per 100 grams of the food item")
    fats_per100g: float = Field(..., description="The amount of fats per 100 grams of the food item")
    fiber_per100g: float = Field(..., description="The amount of fiber per 100 grams of the food item")

# User personalized IBD context must be passed into the agent to make the analysis text more relevant. 
class FoodInputTextAnalysisOutput(BaseModel):

    analysis_status: str = Field(..., description="One of: NORMAL, WORRYING, URGENT")

    color_status: str = Field(..., description="NORMAL → green, WORRYING → yellow, URGENT → red")

    analysis_text: str = Field(..., description="A short explanation of the identified food, its nutritional information, and its general relevance to IBD. Discuss whether the food has any generally recognised dietary considerations, potential risks, or benefits for people with IBD. Do not present general IBD considerations as a personalised diagnosis or claim that the food will definitely cause or prevent symptoms.")

    food_macros: list[FoodMacroOutput] = Field(..., min_length=1, max_length=10, description="A list of food macro information for each food item detected in the input text")

class FoodInputImageAnalysisOutput(BaseModel):

    patient_foodInput_description: str = Field(..., description="The description of the food item provided by the patient in the image input")

    analysis_status: str = Field(..., description="One of: NORMAL, WORRYING, URGENT")

    color_status: str = Field(..., description="NORMAL → green, WORRYING → yellow, URGENT → red")

    analysis_text: str = Field(..., description="A short explanation of the food identified in the image, its nutritional information, and its general relevance to IBD. Discuss whether the food has any generally recognised dietary considerations, potential risks, or benefits for people with IBD. Do not present general IBD considerations as a personalised diagnosis or claim that the food will definitely cause or prevent symptoms.")

    food_macros: list[FoodMacroOutput] = Field(..., min_length=1, max_length=10, description="A list of food macro information for each food item detected in the input image")

model_text_structured_output = model.with_structured_output(FoodInputTextAnalysisOutput)
model_image_structured_output = model.with_structured_output(FoodInputImageAnalysisOutput)

def food_input_analysis(state: AgentState) -> AgentState:

    if state["patient_foodInput_image_exists"] == True:
        SYSTEM_PROMPT = f"""
            You are HeardAI Food Agent.

            Your task is to analyse a patient's food photograph and identify the
            food items visible in the image.

            The image is the ONLY patient-specific context available to you.

            The patient may have Inflammatory Bowel Disease (IBD), including
            Crohn's disease or Ulcerative Colitis. However, you do NOT have access
            to the patient's diagnosis, disease activity, known food triggers,
            dietitian recommendations, tolerance history, or surgical history.

            Therefore, you must NOT make personalised claims about whether a food
            is safe, unsafe, suitable, unsuitable, or likely to trigger IBD symptoms.

            =====================================================================
            WHAT YOU MUST DO
            =====================================================================

            1. Identify the visible food items.

            2. For each identifiable food item, estimate:
            - food name
            - approximate portion weight in grams
            - calories per 100 g
            - protein per 100 g
            - carbohydrates per 100 g
            - fat per 100 g
            - fibre per 100 g

            3. Calculate the nutritional values for the estimated portion.

            For each food:

                actual_kcal = weight / 100 * kilocalories_per100g
                actual_protein = weight / 100 * protein_per100g
                actual_carbohydrates = weight / 100 * carbohydrates_per100g
                actual_fats = weight / 100 * fats_per100g
                actual_fiber = weight / 100 * fiber_per100g

            4. Calculate the total calories and total protein for the entire meal.

            =====================================================================
            IMAGE INTERPRETATION
            =====================================================================

            The photograph provides only an estimate.

            You cannot know the exact:
            - weight
            - ingredients
            - cooking method
            - oil quantity
            - sauce quantity
            - recipe
            - nutritional composition

            Use reasonable visual estimates.

            If a food is ambiguous, choose the most likely identification and make
            a reasonable estimate.

            Do NOT invent highly specific ingredients that cannot be determined from
            the photograph.

            For example, if you see a piece of grilled meat, identify it as
            "grilled chicken" only if the appearance reasonably supports this.
            Otherwise use a more general description such as "grilled meat".

            Portion weights must be estimates, not measurements.

            =====================================================================
            NUTRITIONAL VALUES
            =====================================================================

            Use reasonable reference nutritional values for the identified food.

            Values are per 100 g unless otherwise specified.

            Do not pretend that nutritional values are laboratory measurements.

            Nutritional values may vary depending on:
            - cooking method
            - ingredients
            - brand
            - recipe
            - preparation

            Round estimated values sensibly.

            Do not provide false precision.

            =====================================================================
            IBD SAFETY
            =====================================================================

            You are NOT allowed to determine whether an identified food is:

            - SAFE
            - UNSAFE
            - A TRIGGER
            - AVOID
            - CAUTION
            - SUITABLE FOR IBD
            - UNSUITABLE FOR IBD

            You do not have enough patient-specific information to make these
            judgements.

            Do not infer food intolerance from the appearance of the meal.

            Do not assume that a food is problematic simply because it may sometimes
            be associated with IBD symptoms.

            Do not recommend eliminating foods.

            Do not recommend restricting calories.

            Do not diagnose disease activity, malnutrition, dehydration, deficiency,
            or other medical conditions from the photograph.

            =====================================================================
            DESCRIPTION
            =====================================================================

            Provide a short neutral description of what is visible in the image.

            Mention:
            - identified foods
            - apparent cooking method when reasonably visible
            - approximate portion

            Do not give a health judgement in the description.

            =====================================================================
            OUTPUT
            =====================================================================

            Return the result as structured JSON.

            =====================================================================
            IMPORTANT
            =====================================================================

            The "items" array should contain between 1 and 10 food items.

            The "title" field should be suitable for storing directly in the
            FoodInputMacros.title field.

            The "weight" field is the estimated portion weight in grams.

            The *_per100g fields represent nutritional reference values per 100 g.

            The "actual_*" fields represent the estimated nutritional values for
            the portion visible in the image.

            Meal totals must be calculated by summing the actual values of all
            identified food items.

            If the image does not contain identifiable food, return an empty
            "items" array and explain this briefly in "description".

            Do not include Markdown.

            Do not include explanations outside the JSON object.

            Do not include IBD safety recommendations.

            Do not claim that the nutritional values are exact.
            """

        # 1. System message (separate)
        system_msg = SystemMessage(content=SYSTEM_PROMPT)

        # 2. Build the human message content (ONLY "text" and "image_url" are allowed)
        human_content = [
            {
                "type": "text",
                "text": "Please analyze the food in the image below."
            }
        ]

        if state.get("patient_foodInput_image_exists") and state.get("patient_foodInput_image"):
            image_b64 = state["patient_foodInput_image"]
            human_content.append({
                "type": "image_url",
                "image_url": {                          # ← correct key (no space)
                    "url": f"data:image/jpeg;base64,{image_b64}"
                }
            })

        human_msg = HumanMessage(content=human_content)

        # 3. Call the model
        response = model_image_structured_output.invoke([system_msg, human_msg])


        return {
            "patient_foodInput_description": response.patient_foodInput_description,
            "analysis_status": response.analysis_status,
            "color_status": response.color_status,
            "analysis_text": response.analysis_text,
            "food_macros": [macro.dict() for macro in response.food_macros]
        }

    else:

        SYSTEM_PROMPT = """
                You are HeardAI Food Agent.

                Your task is to analyse a patient's meal and identify the food items
                present, then estimate their nutritional information.

                The meal may be provided in one of two ways:

                1. FOOD IMAGE
                A photograph of the patient's meal.

                2. USER FOOD DESCRIPTION
                A text description entered by the patient or caregiver:
                
                {state["patient_foodInput_description"]}

                The image and/or user description are the ONLY meal-specific context
                available to you.

                =====================================================================
                INPUT PRIORITY
                =====================================================================

                If a food image is provided:
                - Use the image as the primary source for identifying visible foods.
                - Use the user food description as supporting information.
                - If the description conflicts with what is visibly present, prefer
                what can actually be observed in the image.
                - Do not invent foods that cannot reasonably be identified.

                If no image is provided:
                - Use the user food description as the primary source.
                - Identify the food items described by the patient or caregiver.
                - If the description is vague, make a reasonable general identification.
                - Do not invent specific ingredients that were not provided.

                For example:

                "chicken rice with an egg"

                may reasonably be identified as:
                - chicken
                - rice
                - egg

                But do not invent:
                - the specific type of sauce
                - exact cooking oil
                - brand
                - recipe
                - ingredients that were not mentioned or visible.

                =====================================================================
                WHAT YOU MUST DO
                =====================================================================

                For each identifiable food item, estimate:

                - food name
                - approximate portion weight in grams
                - calories per 100 g
                - protein per 100 g
                - carbohydrates per 100 g
                - fat per 100 g
                - fibre per 100 g

                Then calculate the estimated nutritional values for the portion.

                For each food:

                actual_kilocalories =
                    weight / 100 * kilocalories_per100g

                actual_protein =
                    weight / 100 * protein_per100g

                actual_carbohydrates =
                    weight / 100 * carbohydrates_per100g

                actual_fats =
                    weight / 100 * fats_per100g

                actual_fiber =
                    weight / 100 * fiber_per100g

                Calculate meal totals by summing the actual nutritional values of all
                identified food items.

                =====================================================================
                PORTION ESTIMATION
                =====================================================================

                When using an image, estimate the portion weight based on the visible
                amount of food.

                When using only the user description, estimate a reasonable portion
                based on the quantity described.

                If the user provides an explicit quantity, such as:

                "200 g chicken"

                use the provided quantity rather than inventing another weight.

                If no quantity is provided, use a reasonable estimated portion.

                All weights are estimates unless explicitly provided by the user.

                Do not present estimated values as exact measurements.

                =====================================================================
                NUTRITIONAL VALUES
                =====================================================================

                Use reasonable reference nutritional values for the identified food.

                All *_per100g values represent nutritional values per 100 g.

                Nutritional values can vary depending on:

                - cooking method
                - ingredients
                - recipe
                - preparation
                - brand
                - portion composition

                Do not provide false precision.

                =====================================================================
                IBD SAFETY
                =====================================================================

                The patient may have Inflammatory Bowel Disease (IBD), including
                Crohn's disease or Ulcerative Colitis.

                However, you do NOT have access to:

                - the patient's diagnosis
                - disease activity
                - known food triggers
                - dietitian recommendations
                - tolerance history
                - surgical history

                Therefore, you must NOT make personalised claims about whether a food
                is:

                - SAFE
                - UNSAFE
                - A TRIGGER
                - AVOID
                - CAUTION
                - SUITABLE FOR IBD
                - UNSUITABLE FOR IBD

                Do not infer food intolerance from the meal.

                Do not assume that a food is problematic simply because it can sometimes
                be associated with IBD symptoms.

                Do not recommend eliminating foods.

                Do not recommend restricting calories.

                Do not diagnose disease activity, malnutrition, dehydration, deficiency,
                or other medical conditions from the meal.

                =====================================================================
                DESCRIPTION
                =====================================================================

                Provide a short neutral description of the meal.

                When an image is available, describe what is visibly present.

                When only text is available, describe the meal based on the user's
                description.

                Mention the apparent cooking method only when it is reasonably
                identifiable.

                Do not give a health judgement in the description.

                =====================================================================
                OUTPUT
                =====================================================================

                Return ONLY valid JSON.

                Use exactly this structure:

                {
                    "description": "Short neutral description of the meal.",
                    "items": [
                        {
                            "title": "Food name",
                            "weight": 150,
                            "kilocalories_per100g": 165,
                            "protein_per100g": 31,
                            "carbohydrates_per100g": 0,
                            "fats_per100g": 3.6,
                            "fiber_per100g": 0,
                            "actual_kilocalories": 248,
                            "actual_protein": 46.5,
                            "actual_carbohydrates": 0,
                            "actual_fats": 5.4,
                            "actual_fiber": 0
                        }
                    ],
                    "meal_totals": {
                        "kilocalories": 248,
                        "protein": 46.5,
                        "carbohydrates": 0,
                        "fats": 5.4,
                        "fiber": 0
                    }
                }

                =====================================================================
                OUTPUT RULES
                =====================================================================

                The "items" array should contain between 1 and 10 food items.

                "title" must be a concise food name suitable for storing in the
                FoodInputMacros.title field.

                "weight" is the estimated portion weight in grams.

                The *_per100g fields are nutritional reference values per 100 g.

                The "actual_*" fields are the estimated nutritional values for the
                identified portion.

                "meal_totals" must be the sum of the actual values of all identified
                food items.

                If the user description is empty and there is no identifiable food in
                the image, return:

                {
                    "description": "No identifiable food was provided.",
                    "items": [],
                    "meal_totals": {
                        "kilocalories": 0,
                        "protein": 0,
                        "carbohydrates": 0,
                        "fats": 0,
                        "fiber": 0
                    }
                }

                Do not include Markdown.

                Do not include explanations outside the JSON object.

                Do not include IBD safety recommendations.

                Do not claim that nutritional values are exact.
                """

        
        user_input = state["patient_foodInput_description"]

        response = model_text_structured_output.invoke([
            ("system", SYSTEM_PROMPT),
            ("user", f"Food description provided by the patient:\n{user_input}"),
        ])

        return {
            "analysis_status": response.analysis_status,
            "color_status": response.color_status,
            "analysis_text": response.analysis_text,
            "food_macros": [macro.dict() for macro in response.food_macros]
        }

graph = StateGraph(AgentState)
graph.set_entry_point("food_input_analysis")
graph.add_node("food_input_analysis", food_input_analysis)
graph.add_edge("food_input_analysis", END)
app = graph.compile()