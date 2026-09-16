import os
from celery import shared_task
import time

from core.models import (
    # Analysis models
    HeardAIFoodInputAnalysis,
    HeardAIToiletInputAnalysis,
    HeardAIreflectInputAnalysis,
    # Input models
    FoodInputMacros,
    PatientEntry,
    PatientFoodInput,
    PatientReflectInput,
    PatientToiletInput,
    # Core
    PatientRecord,
)

@shared_task
def add(x, y):
    time.sleep(5)  # Simulate a long-running task
    return x + y

@shared_task
def multiply(x, y):
    time.sleep(5)  # Simulate a long-running task
    return x * y

# Revamp of the input tasks 
@shared_task
def patient_reflect_task(
    patient_id: str, 
    reflect_input: str
    ):

    patient = PatientRecord.objects.get(id=patient_id)

    entry = PatientEntry.objects.create(
        patient_record=patient,
        entry_type="REFLECT",
        input_from="PATIENT",
    )

    PatientReflectInput.objects.create(
        patient_entry=entry,
        reflection_text=reflect_input,
    )

@shared_task
def patient_food_task(
    patient_id: str,
    food_input: str
    ):

    patient = PatientRecord.objects.get(id=patient_id)

    entry = PatientEntry.objects.create(
        patient_record=patient,
        entry_type="FOOD",
        input_from="PATIENT",
    )

    PatientFoodInput.objects.create(
        patient_entry=entry,
        food_description=food_input,
    )

@shared_task 
def patient_toilet_task(
    patient_id: str,
    stool_type: str,
    stool_blood: bool,
    stool_urgency: bool,
    stool_at_night: bool,
    ):

    patient = PatientRecord.objects.get(id=patient_id)

    entry = PatientEntry.objects.create(
        patient_record=patient,
        entry_type="TOILET",
        input_from="PATIENT",
    )

    PatientToiletInput.objects.create(
        patient_entry=entry,
        stool_type=stool_type,
        stool_blood=stool_blood,
        stool_urgency=stool_urgency,
        stool_at_night=stool_at_night,
    )

# @shared_task(bind=True, max_retries=3)
@shared_task
def HAI_reflect_analysis_task(entry_id: int, patient_reflect_input: str):
    # Create patient Entry instance 

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

@shared_task
def HAI_toilet_analysis_task(
    entry_id:int,
    patient_stool_type: str,
    patient_stool_blood: str,
    patient_stool_urgency: str,
    patient_stool_at_night: str
):

    from core.agents.HAI_toilet_agent import app
    from core.models import PatientEntry

    # Call the toilet_input_analysis function
    result = app.invoke({
        "patient_stool_type": patient_stool_type,
        "patient_stool_blood": patient_stool_blood,
        "patient_stool_urgency": patient_stool_urgency,
        "patient_stool_at_night": patient_stool_at_night,
        "analysis_status": "",
        "color_status": "",
        "analysis_text": ""
    })

    # Get the patient entry 
    entry = PatientEntry.objects.get(id=entry_id)

    # Get the toilet input 
    toilet_input = entry.toilet_inputs

    ai_analysis = HeardAIToiletInputAnalysis.objects.create(
        patient_toilet_input=toilet_input,
        analysis_status=result['analysis_status'],
        color_status=result['color_status'],
        analysis_text=result['analysis_text']
    )

    ai_analysis.save()
