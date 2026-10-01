from django.urls import path
from core import views
 
urlpatterns = [
    # Patient and Entries summary views 
    path('patient/<uuid:patient_id>', views.PatientRecordDetailAPIView.as_view(), name='patient-detail'),
    path('patient/<uuid:patient_id>/chat', views.PatientHeardAIChatDetailAPIView.as_view(), name='chat-api'),
    path('entry/<int:entry_id>', views.EntryDetailAPIView.as_view(), name='entry-detail'),


    # Getting Patient Detail API
    path('patient/<uuid:patient_id>/detail', views.PatientDetailAPIView.as_view(), name='patient-detail-api'),

    # Creating patient 
    path('patient/CreatePatient/', views.CreatePatientAPIView.as_view(), name='create-patient'),

    # Creating Chat 
    path('patient/chat/create', views.PatientHeardAIChatCreateAPIView.as_view(), name='create-chat'),

    # Heard Agent Daily Summary
    path('patient/daily-summary', views.PatientDailySummaryAgentAPIView.as_view(), name='daily-summary'),

    # Heard Agent Monthly Summary
    path('patient/monthly-summary', views.PatientMonthlySummaryAgentAPIView.as_view(), name='monthly-summary'),
]