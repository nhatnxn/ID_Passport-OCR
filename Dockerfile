FROM python:3.7 AS compile-image
COPY . ./
RUN apt-get update
RUN pip3 install --no-cache-dir --user -r requirements.txt
RUN apt-get install -y libgl1-mesa-dev
EXPOSE 8001
CMD streamlit run --server.port 8001 app.py
