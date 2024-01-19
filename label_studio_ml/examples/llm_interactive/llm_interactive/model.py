import logging
import json
import difflib
import re
import openai
import os
import requests
import pytesseract
import uuid

from PIL import Image
from io import BytesIO
from typing import Union, List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from uuid import uuid4
from tenacity import retry, stop_after_attempt, wait_random

openai.api_key = os.getenv('OPENAI_API_KEY')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


@retry(wait=wait_random(min=5, max=10), stop=stop_after_attempt(6))
def chat_completion_call(messages):
    return openai.ChatCompletion.create(
        model=OpenAIInteractive.OPENAI_MODEL,
        messages=messages,
        n=OpenAIInteractive.NUM_RESPONSES,
        temperature=OpenAIInteractive.TEMPERATURE
    )


def gpt(messages: Union[List[Dict], str]):
    if isinstance(messages, str):
        messages = [{'role': 'user', 'content': messages}]
    logger.debug(f'OpenAI request: {json.dumps(messages, indent=2)}')
    completion = chat_completion_call(messages)
    logger.debug(f'OpenAI response: {json.dumps(completion, indent=2)}')
    response = [choice['message']['content'] for choice in completion.choices]
    # response = ''.join(random.choice(string.ascii_letters) for i in range(50))
    return response


class OpenAIInteractive(LabelStudioMLBase):

    PROMPT_PREFIX = os.getenv('PROMPT_PREFIX', 'prompt')
    USE_INTERNAL_PROMPT_TEMPLATE = bool(int(os.getenv('USE_INTERNAL_PROMPT_TEMPLATE', 1)))
    PROMPT_TEMPLATE = os.getenv('PROMPT_TEMPLATE', '**Source Text**:\n\n"{text}"\n\n**Task Directive**:\n\n"{prompt}"')
    SUPPORTED_INPUTS = ('Image', 'Text', 'HyperText', 'Paragraphs')
    NUM_RESPONSES = int(os.getenv('NUM_RESPONSES', 1))
    TEMPERATURE = float(os.getenv('TEMPERATURE', 0.7))
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4')

    def ocr(self, image_url):
        # Open the image containing the text
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))

        # Run OCR on the image
        text = pytesseract.image_to_string(image)
        return text

    def get_text(self, task, data_type, value_key):
        data = task['data'][value_key]
        if data_type == 'Image':
            return self.ocr(data)
        elif data_type == 'Paragraphs':
            return json.dumps(data)
        else:
            return data

    def get_prompts(self, annotation, prompt_from_name) -> List[str]:
        result = annotation['result']
        for item in result:
            if item.get('from_name') != prompt_from_name:
                continue
            return item['value']['text']
        return []

    def match_choices(self, response: str, original_choices: List[str]) -> List[str]:
        # assuming classes are separated by newlines
        # TODO: support other guardrails
        predicted_classes = response.splitlines()

        matched_labels = []
        for pred in predicted_classes:
            scores = list(map(lambda l: difflib.SequenceMatcher(None, pred, l).ratio(), original_choices))
            matched_labels.append(original_choices[scores.index(max(scores))])
        return matched_labels

    def match_errors(self, original_text, error_text):
        def find_error_position(error_text, source_text):
            start = source_text.find(error_text)
            end = start + len(error_text)
            return start, end

        # Функция для генерации уникального идентификатора
        def generate_id():
            return str(uuid.uuid4())

        # Разбор текста с описаниями ошибок
        error_descriptions = {
            "Fluency errors": [],
            "Grammatical errors": [],
            "Additional input values": [],
            "Missing input values": [],
            "Repetitions": []
        }

        # Изменение регулярного выражения для извлечения текста ошибок
        error_regex = r'<"([^">]+)"'

        # Обработка текста с ошибками
        for category in error_descriptions.keys():
            category_regex = f"{category}:(.*?)(\n#|\Z)"
            category_text = re.search(category_regex, error_text, re.DOTALL)
            if category_text:
                errors = re.findall(error_regex, category_text.group(1), re.DOTALL)
                error_descriptions[category].extend(errors)

        # Поиск позиций ошибок в исходном тексте и формирование JSON
        json_output = []
        for category, errors in error_descriptions.items():
            for error in errors:
                start, end = find_error_position(error, original_text)
                if start != -1:
                    json_output.append({
                        "from_name": "errors",
                        "id": generate_id(),
                        "to_name": "Generation",
                        "type": "labels",
                        "value": {
                            "start": start,
                            "end": end,
                            "text": error,
                            "labels": [category]
                        }
                    })

        return json_output

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        cfg = self.parsed_label_config

        prompt_from_name, prompt_to_name, value_key = self.get_first_tag_occurence(
            # prompt tag
            'TextArea',
            # supported input types
            self.SUPPORTED_INPUTS,
            # if multiple <TextArea> are presented, use one with prefix specified in PROMPT_PREFIX
            name_filter=lambda s: s.startswith(self.PROMPT_PREFIX))
        data_type = cfg[prompt_from_name]['inputs'][0]['type']

        # classification predictor
        use_choices = False
        original_choices = []
        choices_from_name = None
        try:
            choices_from_name, _, _ = self.get_first_tag_occurence(
                'Choices',
                self.SUPPORTED_INPUTS,
                to_name_filter=lambda s: s == prompt_to_name,
            )
            original_choices = cfg[choices_from_name]['labels']
            use_choices = True
        except:
            pass

        # free-form text predictor
        use_textarea = False
        textarea_from_name = None
        try:
            textarea_from_name, _, _ = self.get_first_tag_occurence(
                'TextArea',
                self.SUPPORTED_INPUTS,
                name_filter=lambda s: s != prompt_from_name,
                to_name_filter=lambda s: s == prompt_to_name,
            )
            use_textarea = True
        except:
            pass

        if not use_choices and not use_textarea:
            raise ValueError('No supported tags found: <Choices> or <TextArea>')

        prompts = []
        if context:
            # interactive mode - get prompt from context
            prompts = self.get_prompts(context, prompt_from_name)
        elif prompt := self.get(prompt_from_name):
            # initializing - get existing prompt from storage
            prompts = [prompt]
        prompt = '\n'.join(prompts)

        if not prompts:
            return []

        predictions = []
        base_result = {
            'id': str(uuid4())[:4],
            'from_name': prompt_from_name,
            'to_name': prompt_to_name,
            'type': 'textarea',
            'value': {
                'text': prompts
            }
        }
        model_version = self.model_version
        for task in tasks:
            text = self.get_text(task, data_type, value_key)

            if not prompts:
                response = gpt(text)
            else:
                if self.USE_INTERNAL_PROMPT_TEMPLATE:
                    norm_prompt = self.PROMPT_TEMPLATE.format(text=text, prompt=prompt)
                else:
                    task_data = task['data']
                    norm_prompt = prompt.format(**task_data)
                # run inference
                print('!!!=====> NORM PROMPT ===>', norm_prompt)
                response = gpt(norm_prompt)

            result = []
            if use_choices:
                matched_labels = self.match_choices(response, original_choices)
                result.append({
                    'id': str(uuid4())[:4],
                    'from_name': choices_from_name,
                    'to_name': prompt_to_name,
                    'type': 'choices',
                    'value': {
                        'choices': matched_labels
                    }
                })

            if use_textarea:
                result.append({
                    'id': str(uuid4())[:4],
                    'from_name': textarea_from_name,
                    'to_name': prompt_to_name,
                    'type': 'textarea',
                    'value': {
                        # 'text': [response]
                        'text': response
                    }
                })

            # generate fixed source text using errors found
            errors_found = '\n'.join(response)
            prompt2 = '**Source Text**:\n\n' + text + '\n\n'
            prompt2 += '**Errors Found**:\n\n' + errors_found + '\n\n'
            prompt2 += '**Task Directive**:\n\nFix errors using Errors Found in the Source Text. ' \
                       'Keep text untouched except errors mentioned in Errors Found.'
            response = gpt(prompt2)
            result.append({
                'id': str(uuid4())[:4],
                'from_name': 'fixed_text',
                'to_name': 'Generation',
                'type': 'textarea',
                'value': {
                    'text': response
                }
            })

            # add error highlights in the original source text
            error_highlights = self.match_errors(text, errors_found)
            result += error_highlights

            result.append(base_result)

            predictions.append({'result': result, 'model_version': model_version})

        return predictions

    def prompt_diff(self, old_prompt, new_prompt):
        old_lines = old_prompt.splitlines()
        new_lines = new_prompt.splitlines()
        diff = difflib.unified_diff(old_lines, new_lines, lineterm='')
        return '\n'.join(
                line for line in diff if line.startswith(('+',)) and not line.startswith(('+++', '---')))

    def extract_number(self, s):
        match = re.search(r'^\[(\d+)\]', s)
        return int(match.group(1)) if match else None

    def fit(self, event, data, **additional_params):
        logger.debug(f'Data received: {data}')
        if event not in ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED'):
            return

        prompt_from_name, prompt_to_name, value_key = self.get_first_tag_occurence(
            'TextArea',
            self.SUPPORTED_INPUTS,
            name_filter=lambda s: s.startswith(self.PROMPT_PREFIX))

        prompts = self.get_prompts(data['annotation'], prompt_from_name)
        if not prompts:
            logger.debug(f'No prompts recorded.')
            return

        prompt = '\n'.join(prompts)

        current_model_version = self.get('model_version')
        if current_model_version == 'INITIAL':
            current_prompt = ''
            version_number = -1
            logger.info('No previous model version found.')

        else:
            current_prompt = self.get(prompt_from_name)
            version_number = self.extract_number(current_model_version)
            logger.info(f'Found previous model version {current_model_version}')

        # find substrings that differ between current and new prompt
        # if there are no differences, skip training
        diff = self.prompt_diff(current_prompt, prompt)
        if not diff:
            logger.debug('No prompt diff found.')
            return

        logger.debug(f'Prompt diff: {diff}')
        self.set(prompt_from_name, prompt)

        model_version = f'[{version_number + 1}]{diff}'.strip()
        self.set('model_version', model_version)
        logger.debug(f'Updated model version to {self.get("model_version")}')