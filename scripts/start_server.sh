#!/bin/bash
# Navigate to the bot directory
cd /home/ubuntu/my-bot-with-chart

# Activate the virtual environment
source venv/bin/activate

# Start the bot
nohup python3 Bot_GitHub_V1.py > bot_output.log 2>&1 &
