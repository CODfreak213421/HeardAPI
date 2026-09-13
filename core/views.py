from django.shortcuts import render
from django.utils import timezone
from datetime import timedelta
from rest_framework import generics

# Import models 
from core.models import PatientRecord, PatientEntry, HeardAIMonthlyAnalysis

# import serializers
from core.serializers import PatientDetailEntryAISerializer, PatientEntryCreateFoodInputSerializer, PatientEntryCreateReflectInputSerializer, PatientEntryCreateToiletInputSerializer, PatientRecordSerializer

# Import my celery tasks 
from .tasks import *

# Create your views here.
class PatientRecordDetailAPIView(generics.RetrieveAPIView):
    serializer_class = PatientRecordSerializer
    queryset = PatientRecord.objects.all()
    lookup_field = 'id'
    lookup_url_kwarg = 'patient_id'

class CreatePatientAPIView(generics.CreateAPIView):
    serializer_class = PatientRecordSerializer

class CreatePatientReflectInputAPIView(generics.CreateAPIView):
    serializer_class = PatientEntryCreateReflectInputSerializer

    def perform_create(self, serializer):
        # Save the new PatientEntry instance
        patient_entry = serializer.save()

        # Call the Celery task to analyze the reflection input
        HAI_reflect_analysis_task.delay(
            entry_id=patient_entry.id,
            patient_reflect_input=patient_entry.reflect_inputs.reflection_text
        )

class CreatePatientFoodInputAPIView(generics.CreateAPIView):
    serializer_class = PatientEntryCreateFoodInputSerializer

    def perform_create(self, serializer):
        # Save the new PatientEntry instance
        patient_entry = serializer.save()

        food_input = patient_entry.food_inputs

        if food_input.food_image:
            import base64
            # Convert the image to base64
            with open(food_input.food_image.path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

            HAI_food_analysis_task.delay(
                entry_id=patient_entry.id,
                patient_foodInput_description=food_input.food_description,
                patient_foodInput_image_exists=True,
                patient_foodInput_image=image_base64
            )

        else:
            HAI_food_analysis_task.delay(
                entry_id=patient_entry.id,
                patient_foodInput_description=food_input.food_description,
                patient_foodInput_image_exists=False,
                patient_foodInput_image=""
            )

        


# Create your views here.
class EntryDetailAPIView(generics.RetrieveAPIView):
    serializer_class = PatientDetailEntryAISerializer
    queryset = PatientEntry.objects.all()
    lookup_field = 'id'
    lookup_url_kwarg = 'entry_id'


class CreatePatientToiletInputAPIView(generics.CreateAPIView):
    serializer_class = PatientEntryCreateToiletInputSerializer

    def perform_create(self, serializer):
        # Save the new PatientEntry instance
        patient_entry = serializer.save()

        toilet_input = patient_entry.toilet_inputs

        HAI_toilet_analysis_task.delay(
            entry_id=patient_entry.id,
            patient_stool_type=toilet_input.stool_type,
            patient_stool_blood=toilet_input.stool_blood,
            patient_stool_urgency=toilet_input.stool_urgency,
            patient_stool_at_night=toilet_input.stool_at_night
        )