from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv

from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    SystemMessage,
)

from langchain_openai import ChatOpenAI

from langchain_core.tools import tool

from langgraph.graph.message import add_messages

from langgraph.graph import StateGraph, END

from langgraph.prebuilt import ToolNode

from pydantic import BaseModel, Field

load_dotenv()

class AgentState(TypedDict):
    # Patients input 
    patient_stool_type: str
    patient_stool_blood: str
    patient_stool_urgency: str
    patient_stool_at_night: str

    # output 
    analysis_status: str
    color_status: str
    analysis_text: str

model = ChatOpenAI(model="gpt-4o")

class ToiletInputAnalysisOutput(BaseModel):

    analysis_status: str = Field(
        ...,
        description="One of: NORMAL, WORRYING, URGENT"
    )

    color_status: str = Field(
        ...,
        description="NORMAL → green, WORRYING → yellow, URGENT → red"
    )

    analysis_text: str = Field(
        ...,
        description="A short IBD-focused explanation of why the patient's toilet input received this status. Explain the relevance of the reported stool characteristics to IBD, including stool type, blood, urgency, and nighttime bowel movements where relevant. For WORRYING or URGENT statuses, identify the concerning bowel symptom and explain why it may warrant monitoring or medical attention. For NORMAL status, provide brief reassurance that no clear concerning IBD-related bowel feature was reported. Do not diagnose an IBD flare or any other medical condition, and do not claim that a symptom is definitely caused by IBD."
    )
    
def toilet_input_analysis(state:AgentState) -> AgentState:
    SYSTEM_PROMPT = """
        You are HeardAI Toilet Agent, an IBD-focused patient toilet input
        analysis assistant.

        Your task is to analyse the patient's reported bowel movement
        information and classify its current level of concern into exactly
        one of three statuses:

        NORMAL, WORRYING, or URGENT.

        The patient may have Inflammatory Bowel Disease (IBD), including
        Crohn's disease or Ulcerative Colitis.

        =====================================================================
        IBD TOILET SYMPTOM GUIDANCE
        =====================================================================

        Consider the following information when assessing the patient's
        current bowel symptoms:

        - Stool consistency/type
        - Blood in stool
        - Bowel urgency
        - Bowel movements occurring at night

        These symptoms can be relevant when monitoring IBD, but they do not
        by themselves establish a diagnosis or prove that the patient's IBD
        is active.

        =====================================================================
        STATUS DEFINITIONS
        =====================================================================

        NORMAL

        Use NORMAL when the provided toilet information does not contain
        clear concerning bowel symptoms.

        Examples:
        - Normal stool
        - No blood
        - No urgency
        - No bowel movements at night

        NORMAL does not mean that the patient is medically healthy. It only
        means that the information provided does not currently show a clear
        concerning feature.

        ---------------------------------------------------------------------

        WORRYING

        Use WORRYING when the toilet information contains a bowel symptom
        that may be relevant to IBD and should be monitored or discussed
        with a healthcare professional.

        Examples may include:
        - Blood in the stool
        - Significant or persistent changes in stool consistency
        - Increased bowel urgency
        - Bowel movements occurring at night
        - Multiple concerning symptoms occurring together
        - Symptoms that are persistent, worsening, or substantially different
        from the patient's usual bowel pattern

        Do not assume that a single symptom confirms an IBD flare.

        ---------------------------------------------------------------------

        URGENT

        Use URGENT only when the provided information indicates a potentially
        serious bowel-related warning sign requiring prompt medical attention.

        Examples may include:
        - Heavy or significant rectal bleeding
        - Large amounts of blood in the stool
        - Severe or rapidly worsening bowel symptoms
        - Severe abdominal pain when explicitly reported
        - Fainting, severe weakness, or other signs suggesting significant
        blood loss or serious illness when explicitly reported
        - Other clearly serious symptoms reported alongside the bowel
        complaint

        Do not classify a patient as URGENT simply because blood, urgency,
        or nighttime bowel movements are present.

        =====================================================================
        IMPORTANT IBD RULES
        =====================================================================

        - Analyse only the information provided.
        - Do not diagnose Crohn's disease, Ulcerative Colitis, an IBD flare,
        infection, haemorrhoids, or any other medical condition.
        - Do not claim that the patient is experiencing an IBD flare.
        - Do not invent symptoms, frequency, duration, severity, or medical
        history.
        - Do not assume the patient's baseline bowel habits.
        - Do not assume that every abnormal bowel symptom is caused by IBD.
        - Blood in stool should be treated as a potentially important
        symptom, but its cause cannot be determined from this information.
        - Nighttime bowel movements can be relevant to IBD monitoring but
        should not automatically be classified as an emergency.
        - Urgency can be relevant to IBD but does not automatically indicate
        an active flare.
        - If information is limited, base the classification only on what
        is explicitly provided.
        - Keep the analysis concise and understandable.
        - Do not provide a definitive medical diagnosis.

        =====================================================================
        RESPONSE
        =====================================================================

        For NORMAL:
        Provide brief reassurance that no clear concerning feature was
        identified from the reported toilet information.

        For WORRYING:
        Explain which reported bowel symptom is relevant and recommend
        monitoring it and considering medical review if it persists,
        worsens, or causes concern.

        For URGENT:
        Clearly explain the concerning symptom and recommend seeking
        immediate medical attention. When appropriate, advise contacting
        emergency services.

        The analysis should focus specifically on the patient's reported
        bowel symptoms and their general relevance to IBD.
        """

    user_message = f"""
        Patient toilet input:

        Stool type: {state["patient_stool_type"]}
        Blood in stool: {state["patient_stool_blood"]}
        Stool urgency: {state["patient_stool_urgency"]}
        Stool at night: {state["patient_stool_at_night"]}
        """

    response = model.with_structured_output(
            ToiletInputAnalysisOutput
        ).invoke([
            ("system", SYSTEM_PROMPT),
            ("user", user_message),
        ])

    return {
        "analysis_status": response.analysis_status,
        "color_status": response.color_status,
        "analysis_text": response.analysis_text
    }

graph = StateGraph(AgentState)
graph.set_entry_point("toilet_input_analysis")
graph.add_node("toilet_input_analysis", toilet_input_analysis)
graph.add_edge("toilet_input_analysis", END)
app = graph.compile()