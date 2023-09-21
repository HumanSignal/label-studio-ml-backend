import os
import logging

from transformers import AutoTokenizer, AutoModelForCausalLM
from lxml import etree

from label_studio_ml.model import LabelStudioMLBase

MODEL_NAME = os.getenv('MODEL_NAME')
NUM_RESPONSES = 5
MAX_LENGTH = 1000

logger = logging.getLogger(__name__)


class DialoGPTSimpleGenerator(LabelStudioMLBase):

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def _run_generator(self, texts):
        logger.debug(f'Running generator for texts: {texts}')
        entire_dialog = self.tokenizer.eos_token.join(texts) + self.tokenizer.eos_token

        input_ids = self.tokenizer.encode(entire_dialog, return_tensors='pt')
        logger.debug(f'Got input_ids: {input_ids}')
        # generated a response while limiting the total chat history to 1000 tokens,
        chat_history_ids = self.model.generate(
            input_ids,
            max_length=MAX_LENGTH,
            pad_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=NUM_RESPONSES,
            top_k=NUM_RESPONSES,
            do_sample=True
        )
        logger.debug(f'Got chat_history_ids: {chat_history_ids}')

        responses = []
        for i in range(NUM_RESPONSES):
            response_ids = chat_history_ids[:, input_ids.shape[-1]:][i]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            if response:
                responses.append(response)
            else:
                logger.error(f'Got empty response for indices: {response_ids}')
        logger.debug(f'Got responses: {responses}')
        return responses

    def predict(self, tasks, **kwargs):

        # get <TextArea> tag attributes
        from_name, to_name, value = self.get_first_tag_occurence('TextArea', 'Paragraphs')
        # get <Paragraphs> content keys
        config = etree.fromstring(self.label_config)
        paragraphs = config.find('.//Paragraphs')
        name_key = paragraphs.get('nameKey') or paragraphs.get('namekey') or 'author'
        text_key = paragraphs.get('textKey') or paragraphs.get('textkey') or 'text'

        predictions = []
        for task in tasks:
            dialogue = task['data'][value]
            texts = [item[text_key] for item in dialogue]
            responses = self._run_generator(texts)
            predictions.append({
                'result': [{
                    'from_name': from_name,
                    'to_name': to_name,
                    'type': 'textarea',
                    'value': {
                        'text': responses
                    }
                }]
            })

        return predictions
