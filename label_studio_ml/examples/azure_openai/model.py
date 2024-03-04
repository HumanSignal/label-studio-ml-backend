import logging
import json
import difflib
import re
import openai
import os
import requests
import pytesseract

from uuid import uuid4
from PIL import Image
from io import BytesIO
from typing import Union, List, Dict, Optional, Any
from tenacity import retry, stop_after_attempt, wait_random

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.objects import PredictionValue
from label_studio_sdk.label_interface.object_tags import ImageTag, ParagraphsTag


logger = logging.getLogger(__name__)


@retry(wait=wait_random(min=5, max=10), stop=stop_after_attempt(6))
def chat_completion_call(messages, params, *args, **kwargs):
    # provider="openai", endpoint=None, model="gpt-4",
    # temperature=0.7, num_responses=1, 
    openai.api_key = params.get("api_key",
                                os.getenv('OPENAI_API_KEY', OpenAIInteractive.OPENAI_KEY))
    
    model = params.get("model", OpenAIInteractive.OPENAI_MODEL)
    num   = params.get("num_responses", OpenAIInteractive.NUM_RESPONSES)
    temp  = params.get("temperature", OpenAIInteractive.TEMPERATURE)
    
    model_params = {
        "model": model,
        "messages": messages,
        "n": num,
        "temperature": temp
    }
        
    if params and "provider" in params and \
       params["provider"] == "azure":
        openai.api_type = "azure"
        openai.api_base = params.get("resource_endpoint", None)
        openai.api_version = "2023-05-15"
        model_params["engine"] = params.get("deployment_name")
    
    return openai.ChatCompletion.create(**model_params)

def gpt(messages: Union[List[Dict], str], params, *args, **kwargs):
    """
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    logger.debug(f"OpenAI request: {json.dumps(messages, indent=2)}")
    completion = chat_completion_call(messages, params)

    logger.debug(f"OpenAI response: {json.dumps(completion, indent=2)}")
    response = [ choice["message"]["content"] for choice in completion.choices ]
    
    return response

def extract_number(self, s):
    """
    """
    match = re.search(r'^\[(\d+)\]', s)
    return int(match.group(1)) if match else None

def prompt_diff(self, old_prompt, new_prompt):
    """
    """
    old_lines = old_prompt.splitlines()
    new_lines = new_prompt.splitlines()
    diff = difflib.unified_diff(old_lines, new_lines, lineterm="")
    
    return "\n".join(
        line for line in diff if line.startswith(('+',)) and not line.startswith(('+++', '---')))


class OpenAIInteractive(LabelStudioMLBase):
    """
    """
    OPENAI_KEY = ""
    PROMPT_PREFIX = os.getenv("PROMPT_PREFIX", "prompt")
    USE_INTERNAL_PROMPT_TEMPLATE = bool(int(os.getenv("USE_INTERNAL_PROMPT_TEMPLATE", 1)))
    PROMPT_TEMPLATE = os.getenv("PROMPT_TEMPLATE", '**Source Text**:\n\n"{text}"\n\n**Task Directive**:\n\n"{prompt}"')
    PROMPT_TAG = "TextArea"
    SUPPORTED_INPUTS = ("Image", "Text", "HyperText", "Paragraphs")
    NUM_RESPONSES = int(os.getenv("NUM_RESPONSES", 1))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")

    def _ocr(self, image_url):
        # Open the image containing the text
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))

        # Run OCR on the image
        text = pytesseract.image_to_string(image)
        return text

    def _get_text(self, task_data, object_tag):
        """
        """
        data = task_data.get(object_tag.value_name)
        
        if data is None:
            return None
        
        if isinstance(object_tag, ImageTag):
            return self._ocr(data)
        elif isinstance(object_tag, ParagraphsTag):
            return json.dumps(data)        
        else:
            return data

    def _get_prompts(self, context, prompt_tag) -> List[str]:
        """Getting prompt values
        """
        if context:
            # Interactive mode - get prompt from context
            result = context.get('result')
            for item in result:
                if item.get('from_name') == prompt_tag.name:
                    return item['value']['text']
        # Initializing - get existing prompt from storage
        elif prompt := self.get(prompt_tag.name):            
            return [ prompt ]
        
        return []

    def _match_choices(self, response: List[str], original_choices: List[str]) -> List[str]:
        # assuming classes are separated by newlines
        # TODO: support other guardrails
        matched_labels = []
        predicted_classes = response[0].splitlines()
        
        for pred in predicted_classes:
            scores = list(map(lambda l: difflib.SequenceMatcher(None, pred, l).ratio(), original_choices))
            matched_labels.append(original_choices[scores.index(max(scores))])
        
        return matched_labels
    
    def _find_choices_tag(self, object_tag):
        """Classification predictor
        """
        li = self.label_interface
        
        try:
            choices_from_name, _, _ = li.get_first_tag_occurence(
                'Choices',
                self.SUPPORTED_INPUTS,
                to_name_filter=lambda s: s == object_tag.name,
            )
            
            return li.get_control(choices_from_name)            
        except:
            return None
        
    def _find_textarea_tag(self, prompt_tag, object_tag):
        """Free-form text predictor
        """
        li = self.label_interface
        
        try:
            textarea_from_name, _, _ = li.get_first_tag_occurence(
                'TextArea',
                self.SUPPORTED_INPUTS,
                name_filter=lambda s: s != prompt_tag.name,
                to_name_filter=lambda s: s == object_tag.name,
            )
            
            return li.get_control(textarea_from_name)
        except:
            return None

    def _find_prompt_tags(self):
        """Find prompting tags in the config
        """
        li = self.label_interface
        prompt_from_name, prompt_to_name, value = li.get_first_tag_occurence(
            # prompt tag
            self.PROMPT_TAG,
            # supported input types
            self.SUPPORTED_INPUTS,
            # if multiple <TextArea> are presented, use one with prefix specified in PROMPT_PREFIX
            name_filter=lambda s: s.startswith(self.PROMPT_PREFIX))        
        
        return li.get_control(prompt_from_name), li.get_object(prompt_to_name)
    
    def _validate_tags(self, choices_tag: str, textarea_tag: str) -> None:
        if not choices_tag and not textarea_tag:
            raise ValueError('No supported tags found: <Choices> or <TextArea>')
    
    def _generate_normalized_prompt(self, text: str, prompt: str, task_data: Dict) -> str:
        """
        """
        if self.USE_INTERNAL_PROMPT_TEMPLATE:
            norm_prompt = self.PROMPT_TEMPLATE.format(text=text, prompt=prompt)
        else:
            norm_prompt = prompt.format(**task_data)
            
        return norm_prompt

    def _generate_response_regions(self, response: str, prompt_tag,
                                   choices_tag: str, textarea_tag: str, prompts: List[str]) -> List:
        """
        """
        regions = []
        
        if choices_tag:
            matched_labels = self._match_choices(response, choices_tag.labels)
            regions.append(choices_tag.label(matched_labels))
        
        if textarea_tag:
            regions.append(textarea_tag.label(text=response))
        
        # not sure why we need this but it was in the original code
        regions.append(prompt_tag.label(text=prompts))
        
        return regions
        
    def _predict_single_task(self, task_data: Dict, prompt_tag: Any, object_tag: Any, prompt: str, 
                             choices_tag:str, textarea_tag:str, prompts: List[str]) -> Dict:
        """
        """
        text = self._get_text(task_data, object_tag)
        norm_prompt = self._generate_normalized_prompt(text, prompt, task_data)
        
        # run inference
        # this are params provided through the web interface        
        response = gpt(norm_prompt, self.extra_params)
        regions = self._generate_response_regions(response, prompt_tag, choices_tag, textarea_tag, prompts)
        
        return PredictionValue(result=regions, score=0.1, model_version=str(self.model_version))
        
    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        """
        """
        predictions = []
        
        # prompt tag contains the prompt in the config
        # object tag contains what we plan to label
        prompt_tag, object_tag = self._find_prompt_tags()
        prompts = self._get_prompts(context, prompt_tag)

        if prompts:
            prompt = "\n".join(prompts)
            
            choices_tag = self._find_choices_tag(object_tag)
            textarea_tag = self._find_textarea_tag(prompt_tag, object_tag)
            self._validate_tags(choices_tag, textarea_tag)
            
            for task in tasks:
                task_data = task['data']
                pred = self._predict_single_task(task_data, prompt_tag, object_tag, prompt, 
                                                 choices_tag, textarea_tag, prompts)
                predictions.append(pred)
        
        return ModelResponse(predictions=predictions)
    
    def fit(self, event, data, **additional_params):
        """
        """
        logger.debug(f'Data received: {data}')
        if event not in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED'):
            return
        
        prompt_tag, object_tag = self._find_prompt_tags()
        prompts = self._get_prompts(data['annotation'], prompt_tag)
        
        if not prompts:
            logger.debug(f'No prompts recorded.')
            return

        prompt = '\n'.join(prompts)
        current_prompt = self.get(prompt_tag.name)
               
        # find substrings that differ between current and new prompt
        # if there are no differences, skip training
        diff = self.prompt_diff(current_prompt, prompt)
        if not diff:
            logger.debug('No prompt diff found.')
            return
        
        logger.debug(f'Prompt diff: {diff}')
        self.set(prompt_tag.name, prompt)
        model_version = self.bump_model_version()
        
        logger.debug(f'Updated model version to { str(model_version) }')
        
