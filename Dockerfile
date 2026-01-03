FROM python:3.13

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download open source AdobeRGB profile as we don't want to deal with proprietary Adobe files :)
RUN mkdir -p icc && \
    wget -O icc/AdobeCompat-v4.icc https://github.com/saucecontrol/Compact-ICC-Profiles/raw/refs/heads/master/profiles/AdobeCompat-v4.icc

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
