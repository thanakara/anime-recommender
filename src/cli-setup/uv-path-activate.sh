#!/bin/bash

curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
cd ~/SageMaker/anime-recommender
uv sync
source .venv/bin/activate
