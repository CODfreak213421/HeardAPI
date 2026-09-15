from django.shortcuts import render
from django.utils import timezone
from datetime import timedelta
from rest_framework import generics

# Import models 
from core.models import PatientRecord, PatientEntry, HeardAIMonthlyAnalysis

# import serializers
from core.serializers import PatientDetailEntryAISerializer, PatientRecordSerializer

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

class EntryDetailAPIView(generics.RetrieveAPIView):
    serializer_class = PatientDetailEntryAISerializer
    queryset = PatientEntry.objects.all()
    lookup_field = 'id'
    lookup_url_kwarg = 'entry_id'

# Chat function 
class ChatAPIView(generics.CreateAPIView):
    pass