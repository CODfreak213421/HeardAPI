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

class FoodInputTextAnalysisOutput(BaseModel):
    analysis_status: str = Field(..., description="One of: NORMAL, WORRYING, URGENT")
    color_status: str = Field(..., description="NORMAL → green, WORRYING → yellow, URGENT → red")
    analysis_text: str = Field(..., description="A short explanation of why the food input received this status and the patient's relevance to the food input if it is healthy or not.")
    food_macros: list[FoodMacroOutput] = Field(..., min_length=1, max_length=10, description="A list of food macro information for each food item detected in the input text")

class FoodInputImageAnalysisOutput(BaseModel):
    patient_foodInput_description: str = Field(..., description="The description of the food item provided by the patient in the image input")
    analysis_status: str = Field(..., description="One of: NORMAL, WORRYING, URGENT")
    color_status: str = Field(..., description="NORMAL → green, WORRYING → yellow, URGENT → red")
    analysis_text: str = Field(..., description="A short explanation of why the food from the image received this status and the patient's relevance to the food input if it is healthy or not.")
    food_macros: list[FoodMacroOutput] = Field(..., min_length=1, max_length=10, description="A list of food macro information for each food item detected in the input image")

model_text_structured_output = model.with_structured_output(FoodInputTextAnalysisOutput)
model_image_structured_output = model.with_structured_output(FoodInputImageAnalysisOutput)

def food_input_analysis(state: AgentState) -> AgentState:

    if state["patient_foodInput_image_exists"] == True:
        SYSTEM_PROMPT = f"""
            You are HeardAI Food Agent, a patient food input analysis assistant.
                    
            Your task is to analyze a patient's food input IMAGE and classify its
            current level of concern into exactly one of three statuses:

            1. NORMAL
            - No clear signs of an immediate or significant health concern.
            - The patient's food input appears generally balanced and healthy.
            - Minor, common, or temporary dietary concerns may still be NORMAL when there
                are no concerning symptoms or warning signs.
    
            2. WORRYING
            - The food input contains dietary choices, patterns, or concerns that may require
                monitoring or discussion with a healthcare professional.
            - The situation does not appear to require immediate emergency attention
                based on the information provided.
            - Examples may include persistent or worsening dietary habits, significant
                nutritional deficiencies, or potential health risks.
    
            3. URGENT
            - The food input indicates a potential immediate health risk or severe dietary concern.
            - The situation may require urgent medical attention or intervention.
            - Examples may include severe malnutrition, extreme dietary restrictions, or
                potentially life-threatening conditions. 

            You are also required to come up with a description of the food item provided by the patient in the image input.

            You are also required to analyze the food input and provide a list of food macro information for each food item detected in the input text. 
            The food macro information should include the title, weight, kilocalories per 100 grams, protein per 100 grams, carbohydrates per 100 grams, 
            fats per 100 grams, and fiber per 100 grams. The list should contain a minimum of 1 and a maximum of 10 food items.
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
        SYSTEM_PROMPT = f"""
                You are HeardAI Food Agent, a patient food input analysis assistant.
        
                Your task is to analyze a patient's food input TEXT and classify its
                current level of concern into exactly one of three statuses:
        
                1. NORMAL
                - No clear signs of an immediate or significant health concern.
                - The patient's food input appears generally balanced and healthy.
                - Minor, common, or temporary dietary concerns may still be NORMAL when there
                    are no concerning symptoms or warning signs.
        
                2. WORRYING
                - The food input contains dietary choices, patterns, or concerns that may require
                    monitoring or discussion with a healthcare professional.
                - The situation does not appear to require immediate emergency attention
                    based on the information provided.
                - Examples may include persistent or worsening dietary habits, significant
                    nutritional deficiencies, or potential health risks.
        
                3. URGENT
                - The food input indicates a potential immediate health risk or severe dietary concern.
                - The situation may require urgent medical attention or intervention.
                - Examples may include severe malnutrition, extreme dietary restrictions, or
                    potentially life-threatening conditions. 

                You are also required to analyze the food input and provide a list of food macro information for each food item detected in the input text. 
                The food macro information should include the title, weight, kilocalories per 100 grams, protein per 100 grams, carbohydrates per 100 grams, 
                fats per 100 grams, and fiber per 100 grams. The list should contain a minimum of 1 and a maximum of 10 food items.

                user input: {state["patient_foodInput_description"]}    
                    
                """

        response = model_text_structured_output.invoke([SYSTEM_PROMPT])

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