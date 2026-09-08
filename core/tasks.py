import os
from celery import shared_task
import time

from core.models import HeardAIreflectInputAnalysis, HeardAIFoodInputAnalysis, FoodInputMacros

@shared_task
def add(x, y):
    time.sleep(5)  # Simulate a long-running task
    return x + y

@shared_task
def multiply(x, y):
    time.sleep(5)  # Simulate a long-running task
    return x * y

# @shared_task(bind=True, max_retries=3)
@shared_task
def HAI_reflect_analysis_task(entry_id: int, patient_reflect_input: str):
    from core.agents.HAI_reflect_agent import app
    from core.models import PatientEntry

    # Call the reflect_input_analysis function
    result = app.invoke({
        "patient_reflect_input": patient_reflect_input,
        "analysis_status": "",
        "color_status": "",
        "analysis_text": ""
    })

    # Get the patient entry 
    entry = PatientEntry.objects.get(id=entry_id)

    # Get the reflection input 
    reflection_input = entry.reflect_inputs

    ai_analysis = HeardAIreflectInputAnalysis.objects.create(
        patient_reflect_input=reflection_input,
        analysis_status=result['analysis_status'],
        color_status=result['color_status'],
        analysis_text=result['analysis_text']
    )

    ai_analysis.save()

@shared_task
def HAI_food_analysis_task(
    entry_id:int, 
    patient_foodInput_description: str, 
    patient_foodInput_image_exists: bool, 
    patient_foodInput_image: str):

    from core.agents.HAI_food_agent import app
    from core.models import PatientEntry
    
    # Call the food_input_analysis function
    result = app.invoke({
        "patient_foodInput_description": patient_foodInput_description,
        "patient_foodInput_image_exists": patient_foodInput_image_exists,
        "patient_foodInput_image": patient_foodInput_image,
        "analysis_status": "",
        "color_status": "",
        "analysis_text": "",
        "food_macros": []
    })

    # Get the patient entry 
    entry = PatientEntry.objects.get(id=entry_id)

    # Get the food input 
    food_input = entry.food_inputs

    if patient_foodInput_image_exists:
        food_input.food_description = result['patient_foodInput_description']
        food_input.save()

    ai_analysis = HeardAIFoodInputAnalysis.objects.create(
        patient_food_input=food_input,
        analysis_status=result['analysis_status'],
        color_status=result['color_status'],
        analysis_text=result['analysis_text']
    )

    ai_analysis.save()

    # Save the food macros 
    for macro in result['food_macros']:
        food_macro = FoodInputMacros.objects.create(
            patient_food_input=food_input,
            title=macro['title'],
            weight=macro['weight'],
            kilocalories_per100g=macro['kilocalories_per100g'],
            protein_per100g=macro['protein_per100g'],
            carbohydrates_per100g=macro['carbohydrates_per100g'],
            fats_per100g=macro['fats_per100g'],
            fiber_per100g=macro['fiber_per100g']
        )

        food_macro.save()
