'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-07-19 02:02:18
Version: v1
File: 
Brief: 
'''
'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-06-01 19:28:03
Version: v1
File: 
Brief: 
'''
import os
import time
import textwrap
import ast
from PIL import Image
from IPython.display import display, Markdown

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

import sys
root_dir = '../'
sys.path.append(root_dir)

from utils.lib_io import *
from prompt import *

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True)).data
  
class GEMINI(object):
    def __init__(self,google_api_key,model_name,max_output_tokens=2000,temperature=0.9,system_instruction=None):
        if google_api_key:
            self.google_api_key = google_api_key
        else:
            self.google_api_key = os.environ["GOOGLE_API_KEY"]
        # You should run this in terminal:  
            # for Linux and macOS: export GOOGLE_API_KEY=<YOUR_API_KEY>
            # for Win: set GOOGLE_API_KEY=<YOUR_API_KEY>
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.system_instruction = system_instruction

        self.create_model()
    
    @classmethod
    def init_from_yaml(cls,cfg_path='cfg/cfg_gemini.yaml'):
        cfg = read_yaml_file(cfg_path, is_convert_dict_to_class=True)
        return cls(cfg.google_api_key,cfg.model_name,cfg.max_output_tokens,cfg.temperature,cfg.system_instruction)

    def create_model(self):
        genai.configure(api_key=self.google_api_key)
        
        self.generation_config = genai.GenerationConfig(max_output_tokens=self.max_output_tokens,temperature=self.temperature)
        
        if self.system_instruction:
            self.model = genai.GenerativeModel(model_name=self.model_name,generation_config=self.generation_config,system_instruction=self.system_instruction)
        else:
            self.model = genai.GenerativeModel(model_name=self.model_name,generation_config=self.generation_config)

        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
            }
        # docs about safety settings: https://ai.google.dev/gemini-api/docs/safety-settings
            # Block none	BLOCK_NONE	Always show regardless of probability of unsafe content
            # Block few	BLOCK_ONLY_HIGH	Block when high probability of unsafe content
            # Block some	BLOCK_MEDIUM_AND_ABOVE	Block when medium or high probability of unsafe content
            # Block most	BLOCK_LOW_AND_ABOVE	Block when low, medium or high probability of unsafe content
            # N/A	HARM_BLOCK_THRESHOLD_UNSPECIFIED	Threshold is unspecified, block using default threshold

        self.chat = None

    def get_all_models(self):
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)

    def upload_file(self,uploaded_file_path):
        print(f"Uploading file...")
        uploaded_file = genai.upload_file(path=uploaded_file_path)
        print(f"Completed upload: {uploaded_file.uri}")

        # Check whether the file is ready to be used.
        while uploaded_file.state.name == "PROCESSING":
            print('.', end='')
            time.sleep(10)
            uploaded_file = genai.get_file(uploaded_file.name)
        if uploaded_file.state.name == "FAILED":
            raise ValueError(uploaded_file.state.name)
        
        return uploaded_file

    def list_all_uploaded_files(self):
        for file in genai.list_files():
            print(f"{file.display_name}, URI: {file.uri}")
    
    def delete(self,uploaded_file):
        # Delete an uploaded file.
        genai.delete_file(uploaded_file.name)
        print(f'Deleted file {uploaded_file.uri}')

    def gemini_api(self,prompt,if_p=False,stream=False,timeout=1000):
        response = self.model.generate_content(prompt,safety_settings=self.safety_settings,stream=stream,request_options={"timeout": timeout})
        if if_p:
            print(response.text)
            # print(response.prompt_feedback)
            # print(response.candidates)
            # print(to_markdown(response.text)
        if stream:
            for chunk in response:
                print(chunk.text)
                print("_"*80)
        return response.text

    def text_to_text(self,text,if_p=False):
        response = self.gemini_api(text,if_p=if_p)
        return response
    
    def img_to_text(self,img_path,if_p=False): # PNG/JPEG/WEBP/HEIC/HEIF
        img = Image.open(img_path)
        response = self.gemini_api(img,if_p=if_p)
        return response

    def text_img_to_text(self,text,img_path=None,if_p=False): # can a lot of images, like: [text,img1,img2,img3 ...]
        img = Image.open(img_path)
        response = self.gemini_api([text, img],if_p=if_p)
        return response
    
    def text_imgs_to_text(self,text,img_paths=None,if_p=False): # can a lot of images, like: [text,img1,img2,img3 ...]
        prompt = [text] 
        if img_paths:
            for img_path in img_paths:
                img = Image.open(img_path)
                prompt.append(img) 
        response = self.gemini_api(prompt, if_p=if_p)
        return response
    
    def text_audio_to_text(self,text,audio_path,if_p=False): # WAV/MP3/AIFF/AAC/OGG/FLAC
        audio_file = self.upload_file(audio_path)
        response = self.gemini_api([text,audio_file],if_p=if_p) # "Provide a transcript of the speech from 02:30 to 03:29."
        return response

    def text_video_to_text(self,text,video_path,if_p=False): # /mp4/mpeg/mov/avi/x-flv/mpg/webm/wmv/3gpp
        video_file = self.upload_file(video_path)
        print("Making LLM inference request...")
        response = self.gemini_api([text,video_file],if_p=if_p,timeout=600)
        return response

    def chat_session(self,text,img_path=None,if_p=False):
        if self.chat is None:
            self.chat = self.model.start_chat(history=[])
        if img_path:
            img = Image.open(img_path)
            prompt = [text,img]
        else:
            prompt = text
        response = self.chat.send_message(prompt)
        if if_p:
            print(response.text)
        return response.text

    def chat_history(self):
        for message in self.chat.history:
          print(to_markdown(f'**{message.role}**: {message.parts[0].text}'))

    def count_tokens(self):
        self.model.count_tokens("What is the meaning of life?")
        self.model.count_tokens(self.chat.history)

    def process_response(self,if_p=True):
        import re
        response = re.sub(r'[^a-zA-Z]+', '', response)
        if if_p:
            print(response)
        return response

    def gemini_detection(self,img_path=None,text_prompt=["door that is closed", "door that is open"],if_p=False):
        text = gen_detection_prompt(text_prompt)
        # print(f'text: {text}')
        if img_path:
            img = Image.open(img_path)
        response = self.gemini_api([text, img],if_p=if_p)
        result_probs = ast.literal_eval(response)
        result = 1 if result_probs[1] > result_probs[0] else 0 # Reward is 1 if closer to 'open door' prompt, 0 otherwise
        result_text = text_prompt[result]

        if if_p:
            # print("result:", result)  # prints: 1/0
            print(f'[Gemini INFO] Result: {result_text}\t Label probs: {result_probs}') # door is open/closed
        return result,result_text,result_probs

if __name__ == "__main__":
    gemini = GEMINI.init_from_yaml(f'{root_dir}/cfg/cfg_gemini.yaml')

    ## hl
    # example_img1 = f'{root_dir}/example_img/ll_example_img1.png'
    # last_pmt_action = 'Premove'
    # last_pmt_error = 'SUCCESS'
    # hl_prompt = gen_hl_prompt(last_pmt_action,last_pmt_error)
    # gemini.text_img_to_text(text=hl_prompt,img_path=example_img1,if_p=True)

    ## ll
    example_param1 = [-42.79, 9.61, 94.04]
    example_param2 = [-45.11, 19.86, 2.28]
    example_img1 = f'{root_dir}/example_img/ll_example_img1.png'
    example_img2 = f'{root_dir}/example_img/ll_example_img2.png'
    img_path = f'{root_dir}/trajectory/drawer/fridge3/tjt_015/1.png'
    ll_prompt = gen_ll_prompt(example_param1,example_param2)
    gemini.text_imgs_to_text(ll_prompt,img_paths=[example_img1,example_img2,img_path],if_p=True)

    ## detection
    # example_img1 = f'{root_dir}/example_img/ll_example_img1.png'
    # gemini.gemini_detection(img_path=example_img1,text_prompt=["door that is closed", "door that is open"],if_p=True)