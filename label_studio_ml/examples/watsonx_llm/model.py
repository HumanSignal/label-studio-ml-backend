import label_studio_sdk
import logging
import os
import pytesseract
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from label_studio_sdk.label_interface.control_tags import ControlTag, ObjectTag
from label_studio_sdk.label_interface.object_tags import ImageTag, ParagraphsTag
from label_studio_sdk.label_interface.objects import PredictionValue
from ibm_watsonx_ai.foundation_models import ModelInference
from tqdm import tqdm
from types import SimpleNamespace
from typing import List, Dict, Optional, Tuple, Any

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

logger = logging.getLogger(__name__)


class WatsonXModel(LabelStudioMLBase):
    """
    """
    WATSONX_CREDENTIALS = {
        "url": os.getenv('WATSONX_URL', 'https://us-south.ml.cloud.ibm.com'),
        "apikey": os.getenv('WATSONX_API_KEY')
    }
    LABEL_STUDIO_HOST = os.getenv('LABEL_STUDIO_URL', 'http://localhost:8080')
    LABEL_STUDIO_API_KEY = os.getenv('LABEL_STUDIO_API_KEY')
    PROJECT_ID = os.getenv('WATSONX_PROJECT_ID')
    MODEL_TYPE = os.getenv('WATSONX_MODELTYPE')
    PROMPT_PREFIX = os.getenv('PROMPT_PREFIX', "prompt")
    USE_INTERNAL_PROMPT_TEMPLATE = bool(int(os.getenv("USE_INTERNAL_PROMPT_TEMPLATE", 1)))
    DEFAULT_PROMPT = os.getenv('DEFAULT_PROMPT')
    PROMPT_TEMPLATE = os.getenv('PROMPT_TEMPLATE', '**Source Text**:\n\n"{text}"\n\n**Task Directive**:\n\n"{prompt}"')
    PROMPT_TAG = "TextArea"
    SUPPORTED_INPUTS = ("Image", "Text", "HyperText", "Paragraphs")
    MODEL = None
    def setup(self):
        """Configure any parameters of your model here
        """
        if None in [self.WATSONX_CREDENTIALS["apikey"], self.LABEL_STUDIO_HOST, self.LABEL_STUDIO_API_KEY,
                    self.PROJECT_ID, self.MODEL_TYPE]:
            raise Exception(
                "You must provide your WATSONX credentials, label studio information, and WAtSONX project ID and Model Type in your docker-compose.yml!")
        if self.MODEL_TYPE not in ModelTypes.__members__:
            raise Exception(f"WATSONX_MODELTYPE must be one of {[model for model in ModelTypes.__members__]}")

        # If you want to specify model parameters, you can do so using this empty parameters dictionary
        parameters = {
            GenParams.MAX_NEW_TOKENS: 50,
        }
        if not self.MODEL:
            self.MODEL = ModelInference(
                model_id=ModelTypes[self.MODEL_TYPE],
                params=parameters,
                credentials=self.WATSONX_CREDENTIALS,
                project_id=self.PROJECT_ID
            )
        self.set("model_version", f'{self.__class__.__name__}-v0.0.1')

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
            return [prompt]
        # Default prompt
        elif self.DEFAULT_PROMPT:
            if self.USE_INTERNAL_PROMPT_TEMPLATE:
                logger.error('Using both `DEFAULT_PROMPT` and `USE_INTERNAL_PROMPT_TEMPLATE` is not supported. '
                             'Please either specify `USE_INTERNAL_PROMPT_TEMPLATE=0` or remove `DEFAULT_PROMPT`. '
                             'For now, no prompt will be used.')
                return []
            return [self.DEFAULT_PROMPT]

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

    def _find_prompt_tags(self) -> Tuple[ControlTag, ObjectTag]:
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

    def _generate_normalized_prompt(self, text: str, prompt: str, task_data: Dict, labels: Optional[List[str]]) -> str:
        """
        """
        if self.USE_INTERNAL_PROMPT_TEMPLATE:
            norm_prompt = self.PROMPT_TEMPLATE.format(text=text, prompt=prompt, labels=labels)
        else:
            norm_prompt = prompt.format(labels=labels, **task_data)

        return norm_prompt

    def _generate_response_regions(self, response: List[str], prompt_tag,
                                   choices_tag: ControlTag, textarea_tag: ControlTag, prompts: List[str]) -> List:
        """
        """
        regions = []

        if choices_tag and len(response) > 0:
            matched_labels = self._match_choices(response, choices_tag.labels)
            regions.append(choices_tag.label(matched_labels))

        if textarea_tag:
            regions.append(textarea_tag.label(text=response))

        # not sure why we need this but it was in the original code
        regions.append(prompt_tag.label(text=prompts))

        return regions

    def _predict_single_task(self, task_data: Dict, prompt_tag: Any, object_tag: Any, prompt: str,
                             choices_tag: ControlTag, textarea_tag: ControlTag, prompts: List[str]) -> Dict:
        """
        """
        text = self._get_text(task_data, object_tag)
        # Add {labels} to the prompt if choices tag is present
        labels = choices_tag.labels if choices_tag else None
        norm_prompt = self._generate_normalized_prompt(text, prompt, task_data, labels=labels)

        # run inference
        # this are params provided through the web interface
        full_response = self.MODEL.generate(prompt=norm_prompt)
        response = [full_response['results'][0]['generated_text']]
        print(f'RESPONSE {response}')
        regions = self._generate_response_regions(response, prompt_tag, choices_tag, textarea_tag, prompts)

        return PredictionValue(result=regions, score=0.1, model_version=str(self.model_version))

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
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
