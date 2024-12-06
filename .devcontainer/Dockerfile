FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
    texlive-full \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    biber \
    pandoc \
    git \
    curl && \
    rm -rf /var/lib/apt/lists/*

# 設置工作目錄
WORKDIR /workspace

# 設置容器啟動命令
CMD [ "bash" ]
