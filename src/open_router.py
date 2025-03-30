import os
import requests
import base64
from dotenv import load_dotenv

load_dotenv()

model_name = "mistralai/pixtral-large-2411"
# model_name = "gpt-4o-mini"

class OpenRouter:
    def __init__(self):
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"
        }
        
    def get_response(self, message, image_path=None):
        messages = []
        
        if image_path and os.path.exists(image_path):
            # Read and encode the image
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Create a message with both text and image
            content = [
                {"type": "text", "text": message},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                }
            ]
            messages.append({"role": "user", "content": content})
        else:
            # Text-only message
            messages.append({"role": "user", "content": message})
        
        data = {
            # "model": "gpt-4o-mini",
            "model": model_name,
            "messages": messages
        }
        response = requests.post(self.endpoint, headers=self.headers, json=data)
        return response.json()["choices"][0]["message"]["content"] 
    
open_router = OpenRouter()