# Cat Breed Classification System

## Overview

This is a simple quiz game that allows you to test your knowledge of cat breeds.


## Features
- Local Model Training
  - CNN Model
  - ResNet Model
  - EfficientNet Model
  - VGG Model 
- Remote LLM 
  - ChatGPT-4o-mini (OpenRouter API)

## Installation

```bash
pip install -r requirements.txt
```

## Usage


1. Run the main.py file to start the quiz game.
```bash
python src/main.py
```
2. Select model menu and click each model to build the model.
- Result will saved to .pth file for each model.
3. OpenRouter API key need to be set in environment variable OPENROUTER_API_KEY
4. Select Quiz menu to start the quiz game.
- You can choose to answer the question manually or let the model answer. 
5. Select Visualization menu to see the Grad-CAM visualization of the model.
- You can choose to visualize the result of the model or the custom model.
