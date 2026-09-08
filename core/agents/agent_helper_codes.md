# Generating AI Agent Workflow 

from pathlib import Path

png = app.get_graph().draw_mermaid_png()

output_path = Path.cwd() / "HAI_reflect_agent_flow.png"

with open(output_path, "wb") as f:
    f.write(png)

# HAI Reflect agent custom input 
result = app.invoke({
    "messages": [],
    "patient_reflect_input": "I am feeling good and motivated today",
    "status": "",
    "color_status": "",
    "analysis_text": ""
})

print(result)

# HAI Food agent w/o Image
result = app.invoke({
    "messages": [],
    "patient_foodInput_description": "Chicken Rice with vegetables",
    "patient_foodInput_image_exists": False,
    "patient_foodInput_image": "",
    "status": "",
    "color_status": "",
    "analysis_text": "",
    "food_macros": []
})

print(result)

# HAI Food agent w Image 

## Getting the image base64 
import base64

file_path = "C:/Users/rockv/Documents/SoftwareProjects/HeardAI/media/food_images/food2.jpeg"
with open(file_path, "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

print(image_base64)

## AI Response 
import base64

file_path = "C:/Users/rockv/Documents/SoftwareProjects/HeardAI/media/food_images/food2.jpeg"
with open(file_path, "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")


result = app.invoke({
    "messages": [],
    "patient_foodInput_description": "Analyzing Image...",
    "patient_foodInput_image_exists": True,
    "patient_foodInput_image": image_base64,
    "status": "",
    "color_status": "",
    "analysis_text": "",
    "food_macros": []
})

print(result)