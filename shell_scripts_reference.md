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

