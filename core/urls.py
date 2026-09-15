from django.urls import path
from core import views
 
urlpatterns = [
    path('patient/<uuid:patient_id>', views.PatientRecordDetailAPIView.as_view(), name='patient-detail'),
    path('patient/<uuid:patient_id>/chat', views.PatientHeardAIChatDetailAPIView.as_view(), name='chat-api'),
    path('entry/<int:entry_id>', views.EntryDetailAPIView.as_view(), name='entry-detail'),

    # Creating patient 
    path('patient/CreatePatient/', views.CreatePatientAPIView.as_view(), name='create-patient'),

    # Creating Chat 
    path('patient/chat/create', views.PatientHeardAIChatCreateAPIView.as_view(), name='create-chat'),
]