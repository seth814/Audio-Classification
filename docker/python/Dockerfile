FROM python:3.8
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install \
    gcc nano \
    ffmpeg libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 \
    postgresql postgresql-contrib -y
RUN pip install scipy matplotlib pydub pyaudio psycopg2 numpy flask tensorflow kapre sklearn pandas librosa tqdm wavio
WORKDIR /app
COPY . /app

EXPOSE 5000
# CMD ["python", "-m" , "flask", "run", "--host=0.0.0.0"]
# CMD ["python", "example_docker_postgres.py"]