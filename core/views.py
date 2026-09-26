from django.shortcuts import render
from django.db.models import Q
import json
from datetime import timedelta
from rest_framework import generics
from rest_framework.response import Response

# Import models 
from core.agents import HAI_daily_summary_agent, HAI_monthly_summary_agent
from core.models import PatientRecord, PatientEntry, HeardAIMonthlyAnalysis, HeardAIConversationTrail

# import serializers
from core.serializers import PatientRecordChatSerializer, PatientDetailEntryAISerializer, PatientRecordSerializer, HeardAIConversationTrailSerializer

# Import my celery tasks 
from .tasks import *

# Langchain imports for AI agent interaction
from langchain_core.messages import HumanMessage, AIMessage

# base64 import for image handling
import base64

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

        # Get the patient 
        patient = conversation.patient_record

        # 1. Rebuild history from DB (last 20 messages in chronological order)
        previous = reversed(
            HeardAIConversationTrail.objects
            .filter(patient_record=patient)
            .exclude(id=conversation.id)
            .order_by("-created_at")[:20]
        )

        messages = []
        for trail in previous:
            if trail.response_by == "AIMessage":
                messages.append(AIMessage(content=trail.conversation_text))
            else:
                messages.append(HumanMessage(content=trail.conversation_text))

        # 2. Build the new human message (with optional image)
        content = [{"type": "text", "text": conversation.conversation_text}]

        if getattr(conversation, "Image", None):          # or whatever the field is called
            with conversation.Image.open("rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            mime = getattr(conversation.Image.file, "content_type", "image/jpeg")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"}
            })

        messages.append(HumanMessage(content=content))
        
        # 3. Invoke – the reducer will keep everything
        from core.agents.HAI_Supervisor_agent import app
        result = app.invoke({
            "messages": messages,
            "patient_id": str(patient.id),
            "conversation_id": str(conversation.id)
        })

        ai_message = result["messages"][-1]

        # 4. Persist the AI reply
        HeardAIConversationTrail.objects.create(
            patient_record=patient,
            response_by="AIMessage",
            conversation_text=ai_message.content,
        )

        return Response({"ai_response": ai_message.content})

class PatientDailySummaryAgentAPIView(generics.CreateAPIView):

    def create(self, request, *args, **kwargs):

        patient_id = request.data.get("patient_id")
        date = request.data.get("date")

        # Get all patient entries for the selected day
        # Now using Q filter, I will: 
        # 1. Any entry created on that day (food / reflect / toilet / etc.)
        # 2. Any doctor-appointment entry whose appointment_date matches the day
        queryset = PatientEntry.objects.filter(
            patient_record_id=patient_id,
            ).filter(
                Q(created_at__date=date)
                | Q(doctor_appointment_inputs__appointment_date__date=date)
            ).select_related(
            'food_inputs',
            'reflect_inputs',
            'toilet_inputs',
            'doctor_appointment_inputs',
        ).order_by('created_at')

        # Serialize all entries with their respective AI analysis
        patient_entries = PatientDetailEntryAISerializer(
            queryset,
            many=True
        ).data

        state = {
            "summary": "",
            "one_day_summary_data": json.dumps(
                patient_entries,
                indent=2,
                default=str
            ),
        }

        result = HAI_daily_summary_agent.app.invoke(state)

        return Response({
            "ai_response": result["summary"]
        })

class PatientMonthlySummaryAgentAPIView(generics.CreateAPIView):
    def create(self, request, *args, **kwargs):

        patient_id = request.data.get("patient_id")
        year = request.data.get("year")
        month = request.data.get("month")

        # Get all patient entries for the selected month
        queryset = PatientEntry.objects.filter(
            patient_record_id=patient_id,
            created_at__year=year,
            created_at__month=month,
        ).values_list(
            'created_at__date',
            flat=True
        ).distinct().order_by(
            'created_at__date'
        )

        state = {
            "patient_id": str(patient_id),
            "monthly_summary":"",
            "daily_data":"",
            "list_of_dates": list(queryset),
        }

        result = HAI_monthly_summary_agent.app.invoke(state)

        return Response({
            "ai_response": result["monthly_summary"]
        })