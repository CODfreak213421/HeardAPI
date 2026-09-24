from rest_framework import serializers
from core.models import FoodInputMacros, HeardAIreflectInputAnalysis, HeardAIFoodInputAnalysis, PatientFoodInput, PatientRecord, PatientEntry, PatientReflectInput, PatientToiletInput, HeardAIConversationTrail

# AI Conversation Trail Serializers
class HeardAIConversationTrailSerializer(serializers.ModelSerializer):
    class Meta:
        model = HeardAIConversationTrail
        fields = (
            'id',
            'patient_record',
            'conversation_text',
            'Image',
            'response_by',
            'created_at',
            'updated_at',
        )

class PatientRecordChatSerializer(serializers.ModelSerializer):
    ai_conversation_trails = HeardAIConversationTrailSerializer(
        many=True,
        read_only=True
    )

    class Meta:
        model = PatientRecord
        fields = (
            'id',
            'name',
            'ai_conversation_trails',
        )


# Patient Reflect Input Serializers 
class PatientReflectInputSerializer(serializers.ModelSerializer):
    analysis_status = serializers.CharField(source='ai_analysis.analysis_status', read_only=True)

    class Meta:
        model = PatientReflectInput
        fields = [
            "reflection_text",
            'analysis_status',
        ]

# Patient Food Input Serializers
class PatientFoodInputSerializer(serializers.ModelSerializer):
    analysis_status = serializers.CharField(source='ai_analysis.analysis_status', read_only=True)

    class Meta:
        model = PatientFoodInput
        fields = [
            "food_description",
            "food_image",
            "analysis_status",
        ]

# Patient Toilet Input Serializers
class PatientToiletInputSerializer(serializers.ModelSerializer):
    analysis_status = serializers.CharField(source='ai_analysis.analysis_status', read_only=True)

    class Meta:
        model = PatientToiletInput
        fields = [
            "stool_type",
            "stool_blood",
            "stool_urgency",
            "stool_at_night",
            "analysis_status"
        ]

# Viewing of Full Patient Data in the API
class PatientEntrySerializer(serializers.ModelSerializer):
    subentries = serializers.SerializerMethodField(method_name='get_subentries')

    def get_subentries(self, obj):

        if obj.entry_type == PatientEntry.EntryType.FOOD:
            return PatientFoodInputSerializer(
                obj.food_inputs
            ).data
        elif obj.entry_type == PatientEntry.EntryType.TOILET:
            return PatientToiletInputSerializer(
                obj.toilet_inputs
            ).data

        elif obj.entry_type == PatientEntry.EntryType.REFLECT:
            return PatientReflectInputSerializer(
                obj.reflect_inputs
            ).data

        return None
    
    class Meta:
        model = PatientEntry
        fields = (
            'id',
            'entry_type',
            'subentries',
            'input_from',
            'analysed',
            'created_at',
            'updated_at',
        )
        read_only_fields = ('id', 'created_at', 'updated_at')

class PatientRecordSerializer(serializers.ModelSerializer):
    patient_entries = PatientEntrySerializer(many=True, read_only=True)

    class Meta:
        model = PatientRecord
        fields = (
            'id',
            'name',
            'age',
            'phone_number',
            'email',
            'patient_entries',
            'created_at',
            'updated_at',
        )

        read_only_fields = ('id', 'created_at', 'updated_at')


# Viewing of Detail Entry with AI Analysis 
class HeardAIreflectInputAnalysisSerializer(serializers.ModelSerializer):
    analysis_status = serializers.CharField(source='ai_analysis.analysis_status', read_only=True)
    color_status = serializers.CharField(source='ai_analysis.color_status', read_only=True)
    analysis_text = serializers.CharField(source='ai_analysis.analysis_text', read_only=True)

    class Meta:
        model = PatientReflectInput
        fields = (
            'id',
            'reflection_text',
            'analysis_status',
            'color_status',
            'analysis_text',
        )

class FoodInputMacrosSerializer(serializers.ModelSerializer):
    class Meta:
        model = FoodInputMacros
        fields = (
            'id',
            'title',
            'weight',
            'kilocalories_per100g',
            'protein_per100g',
            'carbohydrates_per100g',
            'fats_per100g',
            'fiber_per100g',
            'created_at',
            'updated_at',
        )

class HeardAIFoodInputAnalysisSerializer(serializers.ModelSerializer):
    analysis_status = serializers.CharField(source='ai_analysis.analysis_status', read_only=True)
    color_status = serializers.CharField(source='ai_analysis.color_status', read_only=True)
    analysis_text = serializers.CharField(source='ai_analysis.analysis_text', read_only=True)

    # nested food macros 
    food_macros = FoodInputMacrosSerializer(many=True, read_only=True)

    class Meta:
        model = PatientFoodInput
        fields = (
            'id',
            'food_description',
            'food_image',
            'analysis_status',
            'color_status',
            'analysis_text',
            'food_macros',
        )

class HeardAIToiletInputAnalysisSerializer(serializers.ModelSerializer):
    analysis_status = serializers.CharField(source='ai_analysis.analysis_status', read_only=True)
    color_status = serializers.CharField(source='ai_analysis.color_status', read_only=True)
    analysis_text = serializers.CharField(source='ai_analysis.analysis_text', read_only=True)

    class Meta:
        model = PatientToiletInput
        fields = (
            'id',
            'stool_type',
            'stool_blood',
            'stool_urgency',
            'stool_at_night',
            'analysis_status',
            'color_status',
            'analysis_text',
        )

class PatientDetailEntryAISerializer(serializers.ModelSerializer):
    subentries = serializers.SerializerMethodField(method_name='get_subentries')

    def get_subentries(self, obj):

        if obj.entry_type == PatientEntry.EntryType.FOOD:
            return HeardAIFoodInputAnalysisSerializer(
                obj.food_inputs
            ).data

        elif obj.entry_type == PatientEntry.EntryType.REFLECT:
            return HeardAIreflectInputAnalysisSerializer(
                obj.reflect_inputs
            ).data
        elif obj.entry_type == PatientEntry.EntryType.TOILET:
            return HeardAIToiletInputAnalysisSerializer(
                obj.toilet_inputs
            ).data

        return None
    
    class Meta:
        model = PatientEntry
        fields = (
            'id',
            'entry_type',
            'subentries',
            'input_from',
            'analysed',
            'created_at',
            'updated_at',
        )
        read_only_fields = ('id', 'created_at', 'updated_at')