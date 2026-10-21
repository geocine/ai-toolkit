FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

LABEL authors="geocine"

# Set noninteractive to avoid timezone prompts
ENV DEBIAN_FRONTEND=noninteractive

# ref https://en.wikipedia.org/wiki/CUDA
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 10.0 12.0"

# Install dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    git \
    curl \
    build-essential \
    cmake \
    wget \
    python3.10 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    python3-venv \
    ffmpeg \
    tmux \
    htop \
    nvtop \
    python3-opencv \
    openssh-client \
    openssh-server \
    openssl \
    rsync \
    unzip \
    nginx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Go
WORKDIR /tmp
RUN curl -LO https://go.dev/dl/go1.21.13.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go1.21.13.linux-amd64.tar.gz && \
    rm go1.21.13.linux-amd64.tar.gz

ENV PATH="/usr/local/go/bin:${PATH}"
ENV GOPATH="/root/go"
ENV PATH="${GOPATH}/bin:${PATH}"

# Install nodejs
WORKDIR /tmp
RUN curl -sL https://deb.nodesource.com/setup_23.x -o nodesource_setup.sh && \
    bash nodesource_setup.sh && \
    apt-get update && \
    apt-get install -y nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set aliases for python and pip
RUN ln -s /usr/bin/python3 /usr/bin/python

# install pytorch before cache bust to avoid redownloading pytorch
RUN pip install --pre --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Fix cache busting by moving CACHEBUST to right before git clone
ARG CACHEBUST=1234
ARG GIT_COMMIT=main
RUN echo "Cache bust: ${CACHEBUST}" && \
    git clone https://github.com/geocine/ai-toolkit.git && \
    cd ai-toolkit && \
    git checkout ${GIT_COMMIT}

WORKDIR /app/ai-toolkit

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --pre --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --force && \
    pip install setuptools==69.5.1 --no-cache-dir

# Build UI-Go (React frontend with Vite)
WORKDIR /app/ai-toolkit/ui-go
RUN npm install && \
    npm run build

# Go setup, Prisma generation, and build
RUN go mod tidy && \
    go run github.com/steebchen/prisma-client-go generate && \
    go run github.com/steebchen/prisma-client-go db push && \
    go build -o /app/ai-toolkit/ui-go/server .

# Configure nginx
COPY nginx.conf /etc/nginx/sites-available/default

# Expose port
EXPOSE 8675

WORKDIR /

COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]

