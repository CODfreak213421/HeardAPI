import uuid

from django.db import models

# Create your models here.
class PatientRecord(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    age = models.IntegerField()
    phone_number = models.CharField(max_length=15)
    email = models.EmailField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

class PatientBackground(models.Model):
    patient_record = models.OneToOneField(PatientRecord, on_delete=models.CASCADE, related_name='background')

    class Diagnosis(models.TextChoices):
        CROHNS = "CROHNS", "Crohn's Disease"
        UC = "UC", "Ulcerative Colitis"
        IBD_UNCLASSIFIED = "IBDU", "IBD Unclassified"

    class DiseaseLocation(models.TextChoices):
        SMALL_BOWEL = "SMALL_BOWEL", "Small Bowel"
        COLON = "COLON", "Colon"
        ILEOCOLONIC = "ILEOCOLONIC", "Ileocolonic"
        UPPER_GI = "UPPER_GI", "Upper GI"
        UNKNOWN = "UNKNOWN", "Unknown"

    diagnosis = models.CharField(max_length=20, choices=Diagnosis.choices, default=Diagnosis.CROHNS)
    diagnosis_date = models.DateField(null=True, blank=True)
    disease_location = models.CharField(max_length=100, choices=DiseaseLocation.choices, blank=True)
    disease_behavior = models.CharField(max_length=100, blank=True)
    surgeries = models.TextField(blank=True)
    family_history = models.TextField(blank=True)
    # Food notes related to the patients IBD triggers 
    # Life Events related to the patient's IBD journey
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Profile for {self.patient_record.name}"

# Patient Entry models for different types of inputs (reflect, toilet, food, medicine)
class PatientEntry(models.Model):
    patient_record = models.ForeignKey(PatientRecord, on_delete=models.CASCADE, related_name='patient_entries')

    class EntryType(models.TextChoices):
        REFLECT = "REFLECT", "Reflect"
        TOILET = "TOILET", "Toilet"
        FOOD = "FOOD", "Food"
        MEDICINE = "MEDICINE", "Medicine"
        FEELING = "FEELING", "Feeling"
        PAIN_TOLERANCE = "PAIN_TOLERANCE", "Pain Tolerance"

    entry_type = models.CharField(max_length=20, choices=EntryType.choices)

    class InputFrom(models.TextChoices):
        PATIENT = "patient", "Patient"
        CAREGIVER = "caregiver", "Caregiver"

    input_from = models.CharField(max_length=20, choices=InputFrom.choices)
    analysed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Input for {self.patient_record.name} at {self.created_at}"

# Patient Reflection models 
class PatientReflectInput(models.Model):
    patient_entry = models.OneToOneField(PatientEntry, on_delete=models.CASCADE, related_name='reflect_inputs')
    reflection_text = models.TextField()

    def __str__(self):
        return f"Reflection for {self.patient_entry.patient_record.name}"

# Patient Food Models
class PatientFoodInput(models.Model):
    patient_entry = models.OneToOneField(PatientEntry, on_delete=models.CASCADE, related_name='food_inputs')
    food_description = models.CharField(max_length=255)
    food_image = models.ImageField(upload_to='food_images/', null=True, blank=True)

    def __str__(self):
        return f"Food Input for {self.patient_entry.patient_record.name}"

# Food Custom model for OpenAI detection 
# https://andrewkushnerov.medium.com/ai-powered-calorie-tracker-how-to-use-chatgpt-and-python-to-analyze-your-meals-e6880a0db4ac
class FoodInputMacros(models.Model):
    patient_food_input = models.ForeignKey(PatientFoodInput, on_delete=models.CASCADE, related_name='food_macros')
    title = models.CharField(max_length=255)
    weight = models.FloatField()
    kilocalories_per100g = models.FloatField()
    protein_per100g = models.FloatField()
    carbohydrates_per100g = models.FloatField()
    fats_per100g = models.FloatField()
    fiber_per100g = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Food Macros for {self.patient_food_input.patient_entry.patient_record.name} at {self.created_at}"

# Patient Feeling Models
class PatientFeelingInput(models.Model):
    patient_entry = models.OneToOneField(PatientEntry, on_delete=models.CASCADE, related_name='feeling_inputs')
    class FeelingType(models.TextChoices):
        HAPPY = "HAPPY", "Happy"
        SAD = "SAD", "Sad"
        ANXIOUS = "ANXIOUS", "Anxious"
        ANGRY = "ANGRY", "Angry"
        TIRED = "TIRED", "Tired"
    feeling_type = models.CharField(max_length=10, choices=FeelingType.choices)

    def __str__(self):
        return f"Feeling Input for {self.patient_entry.patient_record.name}"

# Patient Toilet Models
class PatientToiletInput(models.Model):
    patient_entry = models.OneToOneField(PatientEntry, on_delete=models.CASCADE, related_name='toilet_inputs')
    class StoolType(models.TextChoices):
        NORMAL = "NORMAL", "Normal"
        LOOSE = "LOOSE", "Loose"
        WATERY = "WATERY", "Watery"
        HARD = "HARD", "Hard"
        BLOODY = "BLOODY", "Bloody"
    stool_type = models.CharField(max_length=10, choices=StoolType.choices)
    stool_blood = models.BooleanField(default=False)
    stool_urgency = models.BooleanField(default=False)
    stool_at_night = models.BooleanField(default=False)
    

    def __str__(self):
        return f"Toilet Input for {self.patient_entry.patient_record.name}"

# Patient Medicine input Models
class PatientMedicineInput(models.Model):
    patient_entry = models.OneToOneField(PatientEntry, on_delete=models.CASCADE, related_name='medicine_inputs')
    medicine_name = models.CharField(max_length=100)
    dosage = models.CharField(max_length=50)
    frequency = models.CharField(max_length=50)
    side_effects = models.TextField()

    def __str__(self):
        return f"Medicine Input for {self.patient_entry.patient_record.name}"

# Patient Pain Tolerance input Models
class PatientPainToleranceInput(models.Model):
    patient_entry = models.OneToOneField(PatientEntry, on_delete=models.CASCADE, related_name='pain_tolerance_inputs')
    pain_level = models.IntegerField()
    pain_location = models.CharField(max_length=100)
    pain_description = models.TextField()
    

    def __str__(self):
        return f"Pain Tolerance Input for {self.patient_entry.patient_record.name}"

# AI Agent Analysis Models for different types of Entries (reflect, toilet, food, medicine)
class HeardAIAnalysisBase(models.Model):

    class Status(models.TextChoices):
        NORMAL = "NORMAL", "Normal"
        WORRYING = "WORRYING", "Worrying"
        URGENT = "URGENT", "Urgent"

    analysis_status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.NORMAL
    )

    color_status = models.CharField(
        max_length=20,
        choices=[
            ('green', 'Green'),
            ('yellow', 'Yellow'),
            ('red', 'Red'),
        ]
    )

    analysis_text = models.TextField()

    class Meta:
        abstract = True

class HeardAIreflectInputAnalysis(HeardAIAnalysisBase):
    patient_reflect_input = models.OneToOneField(
        PatientReflectInput,
        on_delete=models.CASCADE,
        related_name='ai_analysis'
    )

    def __str__(self):
        return f"AI Analysis for Reflection of {self.patient_reflect_input.patient_entry.patient_record.name}"

class HeardAIFoodInputAnalysis(HeardAIAnalysisBase):
    patient_food_input = models.OneToOneField(
        PatientFoodInput,
        on_delete=models.CASCADE,
        related_name='ai_analysis'
    )

    def __str__(self):
        return f"AI Analysis for Food Input of {self.patient_food_input.patient_entry.patient_record.name}"

class HeardAIToiletInputAnalysis(HeardAIAnalysisBase):
    patient_toilet_input = models.OneToOneField(
        PatientToiletInput,
        on_delete=models.CASCADE,
        related_name='ai_analysis'
    )

    def __str__(self):
        return f"AI Analysis for Toilet Input of {self.patient_toilet_input.patient_entry.patient_record.name}"

# AI Agent Overall Response from 4 weeks of patient entries
class HeardAIMonthlyAnalysis(models.Model):
    patient_record = models.ForeignKey(PatientRecord, on_delete=models.CASCADE, related_name='ai_analysis')
    patient_entries = models.ManyToManyField(PatientEntry, related_name='ai_analysis_entries')
    analysis_text = models.TextField()
    ibd_symptoms = models.TextField()
    food_sensitivities = models.TextField()
    stress_levels = models.TextField()
    emotional_wellbeing = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"AI Analysis for {self.patient_record.name} at {self.created_at}"