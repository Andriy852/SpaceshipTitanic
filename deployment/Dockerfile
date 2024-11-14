FROM python:3.9.13-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ["pipeline_with_catboost.pkl", "app.py", "helper_functions.py", "imputers.py",  "./"]

EXPOSE 80

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]