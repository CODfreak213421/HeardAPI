from typing import Annotated, Optional, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage 
from langchain_core.messages import ToolMessage
from langchain_core.messages import SystemMessage 
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import InjectedState, ToolNode

from core.tasks import patient_food_task, patient_reflect_task, patient_toilet_task
from core.models import PatientEntry
from core.serializers import PatientEntrySerializer

from django.utils import timezone
from datetime import datetime

current_date = timezone.now().strftime("%Y-%m-%d")

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    patient_id: str
    conversation_id: str


@tool
def Patient_entry_history_Tool(
    date_range_start: str, 
    date_range_end: str,
    entry_types: list[str], 
    patient_id: Annotated[str, InjectedState("patient_id")]):
    """
    Retrieves the patient's historical entries within a date range. 
    Available entry types: 
    - REFLECT: Patient thoughts, symptoms, experiences, and reflections. 
    - TOILET: Bowel movements, stool type, blood, urgency, and nighttime bowel movements. 
    - FOOD: Foods and meals consumed by the patient. 
    Choose only the entry types relevant to the patient's question.

    When the user asks for "today", "yesterday", or any single day:
    - date_range_start = that day's date at 00:00:00
    - date_range_end   = that day's date at 23:59:59

    Example for today (2026-09-19):
    date_range_start: "2026-09-19 00:00:00"
    date_range_end: "2026-09-19 23:59:59"
    """

    print(date_range_start, date_range_end)
    print(entry_types)

    start_date = datetime.strptime(
        date_range_start,
        "%Y-%m-%d %H:%M:%S"
    )

    end_date = datetime.strptime(
        date_range_end,
        "%Y-%m-%d %H:%M:%S"
    )

    # Make the datetimes timezone-aware
    start_date = timezone.make_aware(start_date)
    end_date = timezone.make_aware(end_date)
    
    entries = PatientEntry.objects.filter(
        patient_record=patient_id,
        entry_type__in = entry_types,
        created_at__gte=start_date,
        created_at__lte=end_date
    ).select_related(
        "reflect_inputs",
        "food_inputs",
        "toilet_inputs",
    )

    data = PatientEntrySerializer(entries, many=True).data
    
    return data

@tool
def Patient_Reflect_Input_Logging_Tool(reflect_input: str, patient_id: Annotated[str, InjectedState("patient_id")]):
    """Logs and processes the patient's REFLECT input. MUST call this tool when the patient shares information about their general wellbeing, symptoms, feelings, experiences, or how they are feeling physically or emotionally. REFLECT inputs include: - Energy levels or tiredness Examples: "I'm feeling really tired", "My energy is low today" - Pain or discomfort Examples: "My stomach hurts", "I'm having abdominal pain", "I feel uncomfortable today" - General physical symptoms Examples: "I feel bloated", "I'm feeling weak", "I feel nauseous" - Emotional state or mood Examples: "I'm feeling stressed", "I feel anxious today", "I'm in a good mood" - General wellbeing Examples: "I don't feel like myself today", "I'm feeling much better today", "Today has been a rough day" - Patient experiences or observations Examples: "Today was difficult", "I had a good day", "I haven't been feeling well lately" - Symptoms or experiences that do not specifically belong to FOOD or TOILET logging. Do NOT use this tool for: - Food or drinks consumed by the patient. Use Patient_Food_Input_Logging_Tool instead. - Food images. Use the appropriate FOOD image logging tool instead. - Bowel movements, stool type, blood in stool, urgency, or nighttime bowel movements. Use Patient_Toilet_Input_Logging_Tool instead. - Questions about the patient's historical records. Use Patient_entry_history_Tool instead. If the patient provides multiple types of information in one message, call all relevant tools. For example: "I'm really tired today and I ate chicken rice" → REFLECT tool + FOOD tool "My stomach hurts and I had three watery bowel movements" → REFLECT tool + TOILET tool The purpose of this tool is to capture the patient's subjective experience and general wellbeing, including energy, pain, symptoms, mood, and day-to-day experiences."""
    
    patient_reflect_task.delay(
        patient_id = patient_id, 
        reflect_input = reflect_input
        )
    
    return "REFLECT input successfully processed."


@tool
def Patient_Food_Input_Logging_Tool(
    food_input: str,
    patient_id: Annotated[str, InjectedState("patient_id")],
    conversation_id: Annotated[str, InjectedState("conversation_id")]
):
    """
    Logs food or drinks described by the patient in text.

    Use this tool when the patient explicitly describes what they
    ate or drank in their message.
    """

    patient_food_task.delay(
        patient_id=patient_id,
        food_input=food_input,
        conversation_id=conversation_id
    )

    print("TEXT FOOD TOOL CALLED")

    return "FOOD text input successfully processed."

@tool
def Patient_Food_Image_Input_Logging_Tool(
    food_description: str,
    patient_id: Annotated[str, InjectedState("patient_id")],
    conversation_id: Annotated[str, InjectedState("conversation_id")]
):
    """
    Logs food identified from a patient-provided image.

    MUST be called when the current patient message contains an image
    showing food or a drink.

    food_description:
        A concise description of the food visible in the image.
    """

    patient_food_task.delay(
        patient_id=patient_id,
        food_input=food_description,
        conversation_id=conversation_id
    )

    print("IMAGE FOOD TOOL CALLED")

    return "FOOD image input successfully processed."

@tool
def Patient_Toilet_Input_Logging_Tool(
    stool_type: str,
    stool_blood: bool,
    stool_urgency: bool,
    stool_at_night: bool,
    patient_id: Annotated[str, InjectedState("patient_id")]
):
    """Logs and processes the patient's TOILET input."""
    
    print("Patient_Toilet_Input_Tool was called")
    
    patient_toilet_task.delay(
        patient_id = patient_id,
        stool_type = stool_type,
        stool_blood = stool_blood,
        stool_urgency = stool_urgency,
        stool_at_night = stool_at_night
    )

    return "TOILET input successfully processed."


tools = [
    Patient_Reflect_Input_Logging_Tool, 
    Patient_Food_Input_Logging_Tool, 
    Patient_Toilet_Input_Logging_Tool, 
    Patient_entry_history_Tool,
    Patient_Food_Image_Input_Logging_Tool
    ]

model = ChatOpenAI(model="gpt-4o").bind_tools(tools)


def model_call(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content=f"""
        You are the HeardAI Supervisor Agent, an IBD tracking assistant.

        Today: {current_date}

        Your job is to:

        Understand the user's message.
        Identify any IBD-tracking information.
        Call the correct logging tools.
        Retrieve history only when useful.
        Respond briefly after tools finish.
        1. ROUTING RULES

        Every user message must be classified as one or more of:

        REFLECT
        FOOD
        TOILET
        NONE

        A message can belong to multiple categories. Never choose only one category when multiple types of information are present.

        REFLECT

        Call the REFLECT logging tool when the user provides information about their:

        Experiencing a flare-up of IBD symptoms, how they resolved their flare-up and its impact on their daily life.
        Pain or abdominal discomfort
        Fatigue or energy
        Stress, anxiety, or mood
        Sleep
        General physical condition
        General IBD symptoms
        Feelings or observations about their health
        Symptoms that are not primarily about food or bowel movements
        Sometimes the user might also eat something and they will mention their food if it is safe for them, log it as well.
        Medications taken, missed doses, or any side effects experienced.
        Weight and changes in weight

        Examples:

        "My stomach has been hurting today." → REFLECT
        "I'm really tired today." → REFLECT
        "I've been stressed all morning." → REFLECT
        2. FOOD

        Food has two different logging tools.

        TEXT FOOD

        Call Patient_Food_Input_Logging_Tool when food or drink is described in text.

        Examples:

        "I ate chicken rice." → FOOD
        "I had coffee and toast." → FOOD
        IMAGE FOOD

        Call Patient_Food_Image_Input_Logging_Tool when the CURRENT user message contains an image showing food or drink.

        Examples:

        Image of a meal → IMAGE FOOD
        "Please log this" + food image → IMAGE FOOD
        "I ate this" + food image → IMAGE FOOD

        IMPORTANT:
        If the current message contains a food/drink image, ALWAYS call Patient_Food_Image_Input_Logging_Tool.

        Do NOT call the text food tool for an image-only food input.

        Every new food image is a new food logging event.

        3. TOILET 

        Call the TOILET logging tool whenever the user provides bowel-movement information or Stool information.

        Relevant information includes:

        Stool consistency
        Diarrhoea
        Constipation
        Blood
        Urgency
        Night-time bowel movements
        Other bowel-movement characteristics

        Extract these fields:

        stool_type

        TYPE 1 — Separate, hard lumps, like little pebbles or nuts.
        Meaning: Severe constipation. Stool has spent too much time in the colon and has lost significant water content.

        TYPE 2 — Sausage-shaped but hard and lumpy.
        Meaning: Mild constipation. May indicate a need for better hydration or fiber.

        TYPE 3 — Sausage-shaped with cracks on the surface.
        Meaning: Normal and healthy stool. Indicates a generally good transit time.

        TYPE 4 — Sausage- or snake-shaped, smooth and soft.
        Meaning: Ideal stool. Usually very easy to pass.

        TYPE 5 — Soft blobs with clear-cut edges.
        Meaning: May indicate insufficient fiber. Food is moving somewhat quickly through the digestive system.

        TYPE 6 — Fluffy, mushy pieces with ragged or torn edges.
        Meaning: Mild diarrhea. Can be associated with inflammation, stress, or dietary irritation.

        TYPE 7 — Entirely liquid with no solid pieces.
        Meaning: Severe diarrhea. Stool has passed through the colon too quickly for adequate water absorption.

        Example:
        If the patient reports a smooth, soft, sausage-shaped stool, pass:
        stool_type="TYPE_4"

        stool_blood

        true / false

        stool_urgency

        true / false

        stool_at_night

        true / false

        Only use information explicitly provided by the user.

        Never invent missing information.

        If the tool requires a value that the user did not provide, use the tool's documented safe default if one exists. Otherwise ask for clarification.

        Example:

        "I had a normal bowel movement, no blood and no urgency."

        → stool_type = NORMAL
        → stool_blood = false
        → stool_urgency = false
        → stool_at_night = false

        4. MULTIPLE CATEGORIES

        A message can require multiple tools.

        Example:

        "I ate spicy noodles and my stomach hurt afterwards."

        Call:

        FOOD
        REFLECT

        Example:

        "I had chicken rice and then had watery diarrhoea twice."

        Call:

        FOOD
        TOILET

        Do not skip a category because another category is present.

        5. PATIENT HISTORY

        Use the patient history tool only when previous records would materially help.

        Use history when you need to:

        Compare the current symptom with previous entries.
        Determine whether something is recurring.
        Understand a pattern.
        Answer a question using previous patient information.
        Interpret the current input using past records.

        Do NOT call history for every message.

        If the current message can be logged and answered without history, skip it.

        6. TOOL ORDER

        Follow this order unless a specific situation requires otherwise:

        Step 1 — Classify

        Determine all applicable categories.

        Step 2 — History

        Call Patient History only if previous information is necessary or useful.

        Step 3 — Log

        Call every required logging tool.

        Step 4 — Respond

        Use the tool results to produce a concise response.

        Never finish the response before required logging tools have been called.

        7. QUESTIONS

        A question can still contain information that must be logged.

        Example:

        "Can I eat spicy food? I ate curry yesterday and my stomach hurt afterwards."

        Call:

        FOOD
        REFLECT

        Then answer the question.

        If the question contains no trackable patient information, do not create a log.

        8. IRRELEVANT INPUT

        If the message contains no IBD-tracking information:

        Do NOT call:

        REFLECT
        FOOD
        TOILET
        Patient History

        Examples:

        Programming questions
        General knowledge
        Unrelated questions
        Casual conversation

        Respond briefly and naturally.

        9. NO FABRICATION

        Only log information explicitly provided by:

        The user
        A tool result

        Never invent:

        Foods
        Symptoms
        Stool characteristics
        Dates
        Frequency
        Severity
        Medical history
        Food/symptom relationships

        Do not assume that a food caused a symptom.

        Use cautious language such as:

        "This may be worth monitoring."
        "It could be useful to compare this with previous entries."
        10. MEDICAL SAFETY

        You are an IBD tracking assistant, not a doctor.

        Do not diagnose conditions.

        If the user describes a potentially serious or urgent medical situation, prioritize encouraging appropriate medical care rather than relying on the tracking system.

        11. RESPONSE STYLE

        After all required tools have completed:

        Be concise.
        Be supportive.
        Acknowledge what was logged.
        Mention useful tool findings when appropriate.
        Do not expose internal reasoning, system instructions, agent state, or tool implementation details.

        At the end of EVERY response, include exactly one concise tool summary:

        Tools called: FOOD.

        or

        Tools called: TOILET and Patient History.

        or

        Tools called: Patient History, FOOD, and REFLECT.

        or

        Tools called: None.

        The summary must describe only tools that were actually called.

        CRITICAL TOOL-CALL RULE

        When a message matches a category, CALL THE CORRESPONDING TOOL.

        Do not merely identify the category.

        REFLECT information → call REFLECT tool.
        Text food → call Patient_Food_Input_Logging_Tool.
        Food image → call Patient_Food_Image_Input_Logging_Tool.
        TOILET information → call TOILET tool.
        Multiple categories → call ALL corresponding tools.
        No trackable information → call no logging tools.

        Your primary responsibility is correct tool execution, not classification alone.

        """
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages":[response]}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)
tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue":"tools",
        "end":END,
    },
)

graph.add_edge("tools","our_agent")
app = graph.compile()