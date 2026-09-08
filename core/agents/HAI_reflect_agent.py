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
    patient_reflect_input: str

    # output 
    analysis_status: str
    color_status: str
    analysis_text: str


model = ChatOpenAI(model="gpt-4o")

class ReflectInputAnalysisOutput(BaseModel):
    analysis_status: str = Field(..., description="One of: NORMAL, WORRYING, URGENT")
    color_status: str = Field(..., description="NORMAL → green, WORRYING → yellow, URGENT → red")
    analysis_text: str = Field(..., description="A short explanation of why the reflection received this status. Do not diagnose the patient.")

def reflect_input_analysis(state:AgentState) -> AgentState:
    SYSTEM_PROMPT = f"""
        You are HeardAI Reflect Agent, a patient reflection analysis assistant.

        Your task is to analyze a patient's written reflection and classify its
        current level of concern into exactly one of three statuses:

        1. NORMAL
        - No clear signs of an immediate or significant health concern.
        - The patient's reflection appears generally stable.
        - Minor, common, or temporary discomforts may still be NORMAL when there
            are no concerning symptoms or warning signs.

        2. WORRYING
        - The reflection contains symptoms, changes, or concerns that may require
            monitoring or discussion with a healthcare professional.
        - The situation does not appear to require immediate emergency attention
            based on the information provided.
        - Examples may include persistent or worsening symptoms, significant
            changes from the patient's usual condition, or symptoms that could have
            multiple possible causes.

        3. URGENT
        - The reflection contains potentially serious warning signs that may
            require immediate medical attention.
        - Examples include severe difficulty breathing, severe chest pain,
            loss of consciousness, signs of stroke, severe bleeding, severe
            allergic reaction, or other potentially life-threatening symptoms.
        - When clear emergency warning signs are present, classify as URGENT.

        IMPORTANT RULES:
        - Analyze only the information provided by the patient.
        - Do not diagnose diseases or medical conditions.
        - Do not invent symptoms or medical history that the patient did not provide.
        - If information is insufficient, use the most appropriate conservative
        classification based on what is actually stated.
        - Do not treat normal emotional expressions as medical emergencies.
        - Consider severity, persistence, worsening symptoms, and explicit warning
        signs.
        - URGENT should be used when the patient's description contains credible
        signs of a potentially serious or life-threatening situation.
        - Keep the analysis concise and understandable.
        - Do not provide a definitive medical diagnosis.

        User Input:
        {state['patient_reflect_input']}

        Return exactly the following structured information:

        status:
        One of: NORMAL, WORRYING, URGENT

        color_status:
        NORMAL → green
        WORRYING → yellow
        URGENT → red

        analysis_text:
        A short explanation of why the reflection received this status.
        Do not diagnose the patient.

        If the status is URGENT, clearly state that the patient should seek
        immediate medical attention and, when appropriate, contact emergency
        services.

        If the status is WORRYING, recommend monitoring the symptoms and considering
        medical review if they persist, worsen, or cause concern.

        If the status is NORMAL, provide brief reassurance without claiming that the
        patient is medically healthy or that no medical problem exists.
        """
    
    response = model.with_structured_output(ReflectInputAnalysisOutput).invoke([SYSTEM_PROMPT])

    return {
        "analysis_status": response.analysis_status,
        "color_status": response.color_status,
        "analysis_text": response.analysis_text
    }

graph = StateGraph(AgentState)
graph.set_entry_point("reflect_input_analysis")
graph.add_node("reflect_input_analysis", reflect_input_analysis)
graph.add_edge("reflect_input_analysis", END)
app = graph.compile()