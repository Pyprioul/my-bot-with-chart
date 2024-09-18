#!/bin/bash
# Navigate to the bot directory
cd /home/ubuntu/my-bot-with-chart

# Install Python virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
