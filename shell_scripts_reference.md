# Scripts to run for My server and dependencies

### Django server
python manage.py runserver 

### redis 
Open ubuntu server
run redis-cli 

### celery worker 
celery -A heardapi worker -P threads -E -l info

### Flower
celery -A heardapi flower

### Creating db schema 
python manage.py graph_models core -g --rankdir=RL --output=heard_application_Schema.png

### All my links 
http://127.0.0.1:8000/api/patient/bfdb8574-0419-46a9-a701-83059efaaaf4
http://127.0.0.1:8000/api/patient/bfdb8574-0419-46a9-a701-83059efaaaf4/chat
http://127.0.0.1:8000/api/entry/62

### Save my agent workflow as an image 
from pathlib import Path

graph_png = app.get_graph().draw_mermaid_png()

output_path = Path(__file__).parent / "agent_flow.png"

with open(output_path, "wb") as f:
    f.write(graph_png)

### Overriding created at date field 
from yourapp.models import PatientEntry
from django.utils import timezone
import datetime

entry = PatientEntry.objects.get(id=123)      
entry.created_at = timezone.make_aware(datetime.datetime(2024, 5, 10, 14, 30))
entry.save(update_fields=["created_at"])