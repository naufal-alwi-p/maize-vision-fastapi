FROM python:3.13.12

ARG CONVNEXT_FILE_ID
ARG MAXVIT_FILE_ID

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

# Source - https://stackoverflow.com/a/63377623
# Posted by Tushar Kolhe, modified by community. See post 'Timeline' for change history
# Retrieved 2026-05-13, License - CC BY-SA 4.0
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --no-cache-dir -r /code/requirements.txt

COPY ./download_model_weights.py /code/download_model_weights.py

RUN python /code/download_model_weights.py --convnext-file-id $CONVNEXT_FILE_ID --maxvit-file-id $MAXVIT_FILE_ID

COPY ./app /code/app

CMD ["fastapi", "run", "app/main.py", "--port", "7860"]
