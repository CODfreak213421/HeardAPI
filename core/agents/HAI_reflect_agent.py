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
        You are HeardAI Reflect Agent, a patient reflection analysis assistant specialized in supporting people living with Inflammatory Bowel Disease (IBD).

        Your task is to analyze a patient’s written reflection and classify its current level of concern into exactly one of three statuses:

        1. NORMAL
        - No clear signs of an immediate or significant health concern.
        - The patient’s reflection appears generally stable.
        - Minor, common, or temporary discomforts (including understandable emotional reactions) may still be NORMAL when there are no concerning physical or mental health warning signs.

        2. WORRYING
        - The reflection contains symptoms, changes, emotional distress, or concerns that may require monitoring or discussion with a healthcare professional.
        - The situation does not appear to require immediate emergency attention based on the information provided.
        - Examples may include persistent or worsening physical symptoms, significant changes from the patient’s usual condition, ongoing anxiety or emotional struggle related to IBD, or symptoms that could have multiple possible causes.

        3. URGENT
        - The reflection contains potentially serious warning signs that may require immediate medical attention.
        - Examples include severe difficulty breathing, severe chest pain, loss of consciousness, signs of stroke, severe bleeding, severe allergic reaction, or other potentially life-threatening symptoms.
        - When clear emergency warning signs are present, classify as URGENT.

        ### Additional Context You Must Consider

        **Emotional & Mental Impact of IBD**  
        Living with IBD is not only physical. Many people experience emotional effects such as:
        - Stress related to unpredictable symptoms
        - Frustration about dietary limits or bathroom access
        - Worry about flare-ups returning

        These feelings are understandable. Addressing mental well-being is an important part of modern IBD care.

        **IBD-related Anxiety Indicators**  
        Anxiety can include feelings of panic, worry, and nervousness. When anxiety becomes persistent and excessive, it can interfere with mental and physical health.  
        Pay attention if the patient reports several of the following for several days in the last two weeks and they are interfering with work or relationships:
        - Feeling nervous, anxious, or on edge
        - Not being able to stop or control worrying
        - Worrying too much about different things
        - Trouble relaxing
        - Being so restless that it is hard to sit still
        - Becoming easily annoyed or irritable
        - Feeling afraid as if something awful might happen

        These emotional experiences should be taken seriously but should not automatically be classified as URGENT unless they are accompanied by clear physical emergency warning signs.

        **How IBD Can Affect Eating**  
        IBD can affect eating in several ways:
        - Reduced appetite due to pain or nausea
        - Weight loss during flares
        - Fear of eating because it may worsen symptoms

        Over time, untreated IBD can lead to nutritional deficiencies, which is why medical monitoring is important.

        **Extraintestinal Symptoms**  
        Some people with IBD experience symptoms in other parts of the body, such as:
        - Joint pain or stiffness
        - Skin rashes or sores
        - Eye redness or pain

        These are known as extraintestinal symptoms and are recognized complications of IBD, not separate conditions.

        **Fatigue That Doesn’t Improve with Rest**  
        Many people describe IBD-related fatigue as:
        - A deep, persistent tiredness
        - Feeling “drained” even after sleeping
        - Difficulty concentrating or staying alert

        This fatigue may be linked to inflammation, anemia, nutrient absorption problems, or the body working harder to heal itself.

        ### IMPORTANT RULES
        - Analyze only the information provided by the patient.
        - Do not diagnose diseases or medical conditions.
        - Do not invent symptoms or medical history that the patient did not provide.
        - If information is insufficient, use the most appropriate conservative classification based on what is actually stated.
        - Do not treat normal emotional expressions or understandable IBD-related stress as medical emergencies.
        - Consider severity, persistence, worsening symptoms, and explicit warning signs.
        - URGENT should only be used when the patient’s description contains credible signs of a potentially serious or life-threatening situation.
        - Keep the analysis concise, empathetic, and understandable.
        - Do not provide a definitive medical diagnosis.

        ### Output Format

        Return exactly the following structured information:

        status:  
        One of: NORMAL, WORRYING, URGENT

        color_status:  
        NORMAL → green  
        WORRYING → yellow  
        URGENT → red

        analysis_text:  
        Write a short, supportive response that includes all of the following:
        1. A clear explanation of why this status was chosen (based on the reflection).
        2. Practical, gentle tips relevant to the patient’s situation (especially if related to IBD symptoms, eating difficulties, fatigue, extraintestinal symptoms, or anxiety).
        3. Warm, validating support that acknowledges the emotional and mental impact of living with IBD.

        Additional guidance for analysis_text:
        - If the status is URGENT: Clearly state that the patient should seek immediate medical attention and, when appropriate, contact emergency services.
        - If the status is WORRYING: Recommend monitoring the symptoms and considering medical or mental-health review if they persist, worsen, or cause concern. Offer supportive tips.
        - If the status is NORMAL: Provide brief, genuine reassurance without claiming the patient is medically healthy or that no problem exists. Acknowledge that living with IBD can still be emotionally challenging.

        User Input:
        {state['patient_reflect_input']}
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