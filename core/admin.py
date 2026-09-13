from django.contrib import admin
from core.models import PatientBackground, PatientRecord, PatientEntry

# Register your models here.
@admin.register(PatientRecord)
class PatientRecordAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'email', 'created_at', 'updated_at')
    search_fields = ('name', 'email')
    list_filter = ('created_at', 'updated_at')

admin.site.register(PatientBackground)
admin.site.register(PatientEntry)