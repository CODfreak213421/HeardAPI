from django.shortcuts import render
from django.utils import timezone
from datetime import timedelta
from rest_framework import generics
from rest_framework.response import Response

# Import models 
from core.models import PatientRecord, PatientEntry, HeardAIMonthlyAnalysis, HeardAIConversationTrail

# import serializers
from core.serializers import PatientRecordChatSerializer, PatientDetailEntryAISerializer, PatientRecordSerializer, HeardAIConversationTrailSerializer

# Import my celery tasks 
from .tasks import *

# Langchain imports for AI agent interaction
from langchain_core.messages import HumanMessage, AIMessage

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

# Chat Views for list and function 
class PatientHeardAIChatDetailAPIView(generics.RetrieveAPIView):
    serializer_class = PatientRecordChatSerializer
    queryset = PatientRecord.objects.all()
    lookup_field = 'id'
    lookup_url_kwarg = 'patient_id'

class PatientHeardAIChatCreateAPIView(generics.CreateAPIView):
    serializer_class = HeardAIConversationTrailSerializer

    def create(self, request, *args, **kwargs):

        # Validate the incoming request data
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        # Save the new conversation entry
        conversation = serializer.save()          

        # 1. Rebuild history from DB
        previous = (
            HeardAIConversationTrail.objects
            .filter(patient_record=conversation.patient_record)
            .order_by("created_at")               
            .exclude(id=conversation.id)
        )

        messages = []
        for trail in previous:
            if trail.response_by == "AIMessage":
                messages.append(AIMessage(content=trail.conversation_text))
            else:
                messages.append(HumanMessage(content=trail.conversation_text))

        # 2. Add the brand-new human message
        messages.append(HumanMessage(content=conversation.conversation_text))
        
        # 3. Invoke – the reducer will keep everything
        from core.agents.HAI_Supervisor_agent import app
        result = app.invoke({"messages": messages})

        ai_message = result["messages"][-1]

        # 4. Persist the AI reply
        HeardAIConversationTrail.objects.create(
            patient_record=conversation.patient_record,
            response_by="AIMessage",
            conversation_text=ai_message.content,
        )

        return Response({"ai_response": ai_message.content})

