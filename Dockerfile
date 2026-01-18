FROM python:3.13

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xinput0 \
    libxcb-xfixes0 \
    libxcb-shape0 \
    libxcb-cursor0 \
    libdbus-1-3 \
    libegl1 \
    libfontconfig1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download ICC Profiles
RUN mkdir -p icc && \
    BASE_URL="https://github.com/saucecontrol/Compact-ICC-Profiles/raw/refs/heads/master/profiles" && \
    wget -O icc/AdobeCompat-v4.icc "$BASE_URL/AdobeCompat-v4.icc" && \
    wget -O icc/sRGB-v4.icc "$BASE_URL/sRGB-v4.icc" && \
    wget -O icc/ProPhoto-v4.icc "$BASE_URL/ProPhoto-v4.icc" && \
    wget -O icc/DisplayP3-v4.icc "$BASE_URL/DisplayP3-v4.icc" && \
    wget -O icc/Rec2020-v4.icc "$BASE_URL/Rec2020-v4.icc" && \
    wget -O icc/WideGamut-v4.icc "$BASE_URL/WideGamut-v4.icc"

COPY . .

ENTRYPOINT ["python", "desktop.py"]
