import logging
import difflib
import os

from io import BytesIO
from typing import Union, List, Dict, Optional, Any, Tuple

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.label_interface.objects import PredictionValue
from label_studio_sdk.label_interface.control_tags import ControlTag, ObjectTag

from langchain_chroma import Chroma
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)
import nltk


rag_chain = None
last_prompts = None

def iife_rag():
    global rag_chain
    if rag_chain:
        return rag_chain

    nltk.download('punkt')
    llm = ChatOpenAI(model=RagQuickstart.OPENAI_MODEL)

    loader = DirectoryLoader('/data/label-studio/docs/source', glob="**/*.md")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know."
        "\n\n"
        "Also consider this additional context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain


def get_rag_response(messages: Union[List[Dict], str], params, *args, **kwargs):
    """
    """
    logger.debug(f"OpenAI request: {messages}, params={params}")
    rag = iife_rag()
    response = rag.invoke({"input": messages})
    print(f'rag response: {response=} {response["answer"]=} {response["context"]=}')
    return response #[]

def get_evaluation(response) -> Tuple[str, float]:
    llm = ChatOpenAI(model=RagQuickstart.OPENAI_MODEL)
    messages = [
        ("system", "Evaluate the following answer to the given question (along with any additional context provided), both for correctness as well as whether it is supported by the given documentation."),
        ("human", "QUESTION AND CONTEXT: " + response["input"]),
        ("human", "ANSWER:" + response["answer"]),
        ("human", "DOCUMENTATION: " + str(response["context"])),
    ]
    evaluation = llm.invoke(messages)
    print(f'\n\n{evaluation=}\n\n')

    float_eval = None
    attempt = 0
    while float_eval is None and attempt <= 3:
        attempt += 1
        print(f'attempt {attempt} to get numerical evaluation')
        numerical_eval_messages = [
            ("system", "Respond with ONLY a float between 0 and 1 inclusive, that represents the quality of the answer according to the given evaluation. 0 means the answer is completely wrong and unsupported by documentation, 1 means the answer is completely correct and supported by documentation."),
            ("human", "EVALUATION: " + evaluation.content),
        ]
        numerical_evaluation = llm.invoke(numerical_eval_messages)
        print(f'\n\n{numerical_evaluation=}\n\n')

        try:
            float_eval = float(numerical_evaluation.content)
        except ValueError:
            print('Could not convert numerical evaluation to float', numerical_evaluation.content)

        if float_eval < 0:
            float_eval = 0.0
        elif float_eval > 1:
            float_eval = 1.0

    return evaluation.content, float_eval

class RagQuickstart(LabelStudioMLBase):
    """
    """
    last_prompts = None
    OPENAI_PROVIDER = os.getenv("OPENAI_PROVIDER", "openai")
    OPENAI_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo-0125')
    PROMPT_PREFIX = os.getenv("PROMPT_PREFIX", "prompt")
    USE_INTERNAL_PROMPT_TEMPLATE = bool(int(os.getenv("USE_INTERNAL_PROMPT_TEMPLATE", 1)))
    # if set, this prompt will be used at the beginning of the session
    DEFAULT_PROMPT = os.getenv('DEFAULT_PROMPT')
    # if set, additional context will be cleared between tasks
    CLEAR_CONTEXT_BETWEEN_TASKS = bool(int(os.getenv("CLEAR_CONTEXT_BETWEEN_TASKS", 1)))
    PROMPT_TEMPLATE = os.getenv("PROMPT_TEMPLATE", '**Question to answer**:\n\n"{text}"\n\n**Additional context (may be empty)**:\n\n"{prompt}"')
    PROMPT_TAG = "TextArea"
    SUPPORTED_INPUTS = ("Text",)

    def setup(self):
        pass

    def _get_text(self, task_data, object_tag):
        """
        """
        print(f'{task_data=}, {object_tag=}')
        data = task_data.get(object_tag.value_name)
        return data

    def _get_prompts(self, context, prompt_tag) -> List[str]:
        """Getting prompt values
        """
        if context:
            print('going down context branch', context)
            # Interactive mode - get prompt from context
            result = context.get('result')
            for item in result:
                if item.get('from_name') == prompt_tag.name:
                    return item['value']['text']
        # Initializing - get existing prompt from storage
        elif prompt := self.get(prompt_tag.name):
            print('going down prompt branch', prompt)
            print('context:', context)
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

    def _validate_tags(self, textarea_tag: str) -> None:
        if not textarea_tag:
            raise ValueError('No supported tags found: <TextArea>')

    def _generate_normalized_prompt(self, text: str, prompt: str, task_data: Dict) -> str:
        """
        """
        if self.USE_INTERNAL_PROMPT_TEMPLATE:
            norm_prompt = self.PROMPT_TEMPLATE.format(text=text, prompt=prompt)
        else:
            norm_prompt = prompt.format(**task_data)

        return norm_prompt

    def _generate_response_regions(self, response: List[str], prompt_tag, textarea_tag: ControlTag, prompts: List[str]) -> List:
        """
        """
        regions = []

        if textarea_tag:
            regions.append(textarea_tag.label(text=response))

        # not sure why we need this but it was in the original code
        regions.append(prompt_tag.label(text=prompts))

        return regions

    def _predict_single_task(self, task_data: Dict, prompt_tag: Any, object_tag: Any, prompt: str,
                             textarea_tag: ControlTag, prompts: List[str]) -> Dict:
        """
        """
        text = self._get_text(task_data, object_tag)
        norm_prompt = self._generate_normalized_prompt(text, prompt, task_data)
        response = get_rag_response(norm_prompt, self.extra_params)
        evaluation, float_eval = get_evaluation(response)

        regions = self._generate_response_regions([response["answer"] + f"\n\nEVALUATION (score: {float_eval})\n\n" + evaluation + "\n\nCONTEXT\n\n" + str(response["context"])], prompt_tag, textarea_tag, prompts)

        return PredictionValue(result=regions, score=(float_eval or 0.1), model_version=str(self.model_version))

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """
        """
        global last_prompts
        predictions = []

        # prompt tag contains the prompt in the config
        # object tag contains what we plan to label
        print('\n\nFINDING PROMPT TAGS\n\n')
        prompt_tag, object_tag = self._find_prompt_tags()
        prompts = self._get_prompts(context, prompt_tag)

        if self.CLEAR_CONTEXT_BETWEEN_TASKS:
            # hack: let's say if there are no predictions, then there are no prompts
            if not tasks[0].get('predictions'):
                prompts = []
        else:
            if not tasks[0].get('predictions') and last_prompts:
                prompts = last_prompts
        last_prompts = prompts


        print('\n\nPROMPTS\n\n', prompts)
        prompt = "\n".join(prompts)

        textarea_tag = self._find_textarea_tag(prompt_tag, object_tag)
        self._validate_tags(textarea_tag)

        for task in tasks:
            # preload all task data fields, they are needed for prompt
            task_data = self.preload_task_data(task, task['data'])
            pred = self._predict_single_task(task_data, prompt_tag, object_tag, prompt,
                                                textarea_tag, prompts)
            predictions.append(pred)

        return ModelResponse(predictions=predictions)
