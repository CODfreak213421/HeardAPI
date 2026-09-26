from django.contrib import admin
from core.models import HeardAIConversationTrail, PatientBackground, PatientChampionStats, PatientRecord, PatientEntry

# Register your models here.
@admin.register(PatientRecord)
class PatientRecordAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'email', 'created_at', 'updated_at')
    search_fields = ('name', 'email')
    list_filter = ('created_at', 'updated_at')

admin.site.register(PatientBackground)
admin.site.register(HeardAIConversationTrail)
admin.site.register(PatientChampionStats)

@admin.register(PatientEntry)
class PatientEntryAdmin(admin.ModelAdmin):
    list_display = ("id", "patient_record", "entry_type", "created_at")
    search_fields = ("entry_type",)
    list_filter = ("created_at", "entry_type")