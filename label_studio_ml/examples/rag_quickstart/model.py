import logging
import os

from typing import Union, List, Dict, Optional, Any, Tuple

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.label_interface.objects import PredictionValue
from label_studio_sdk.label_interface.control_tags import ControlTag, ObjectTag

from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate
from langchain_chroma import Chroma
from langchain_core.documents import Document
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

    nltk.download("punkt")
    nltk.download("averaged_perceptron_tagger")
    llm = ChatOpenAI(model=RagQuickstart.OPENAI_RAG_MODEL)

    loader = DirectoryLoader(
        RagQuickstart.DOCUMENTATION_PATH, glob=RagQuickstart.DOCUMENTATION_GLOB
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(model=RagQuickstart.OPENAI_EMBEDDING_MODEL),
    )
    retriever = vectorstore.as_retriever()

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved documentation to answer "
        "the question. If you don't know the answer, say that you "
        "don't know."
        "\n\n"
        "{context}"
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


def format_documentation(documentation: List[Document]) -> List[str]:
    """Format documentation for display in a TextArea"""

    return [
        f"source: {d.metadata.get('source', 'source unknown')}\n\n{d.page_content}"
        for d in documentation
    ]


def get_rag_response(messages: Union[List[Dict], str], params, *args, **kwargs):
    """ """
    logger.debug(f"rag request: {messages}, params={params}")
    rag = iife_rag()
    response = rag.invoke({"input": messages})
    print(f'rag response: {response=} {response["answer"]=} {response["context"]=}')
    return response


def get_ragas_evaluation(response) -> Tuple[List[str], float]:
    ragas_dataset = {
        "question": [response["input"]],
        "answer": [response["answer"]],
        "contexts": [format_documentation(response["context"])],
    }
    dataset = Dataset.from_dict(ragas_dataset)
    score = evaluate(dataset, metrics=[faithfulness, answer_relevancy])

    results = []
    for item in score.items():
        rounded = "{:.2f}".format(round(item[1], 2))
        results.append(f"{item[0]}: {rounded}")

    return results, (score["faithfulness"] + score["answer_relevancy"]) / 2


def get_textual_evaluation(response) -> Tuple[str, float]:
    llm = ChatOpenAI(model=RagQuickstart.OPENAI_EVALUATION_MODEL)
    messages = [
        (
            "system",
            "Evaluate the following answer to the given question (along with any additional instructions provided), "
            "both for correctness as well as whether it is supported by the given documentation. All claims about "
            "documentation must cite specific documentation that was provided, along with the path to the source. Do "
            "not use first person or provide a revised answer.",
        ),
        ("human", "QUESTION AND ADDITIONAL INSTRUCTIONS: " + response["input"]),
        ("human", "ANSWER: " + response["answer"]),
        ("human", "DOCUMENTATION: " + str(response["context"])),
    ]
    evaluation = llm.invoke(messages)
    print(f"\n\n{evaluation=}\n\n")

    return evaluation.content


class RagQuickstart(LabelStudioMLBase):
    """ """

    DOCUMENTATION_PATH = os.getenv("DOCUMENTATION_PATH")
    DOCUMENTATION_GLOB = os.getenv("DOCUMENTATION_GLOB")
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_EMBEDDING_MODEL = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    OPENAI_RAG_MODEL = os.getenv("OPENAI_RAG_MODEL", "gpt-3.5-turbo-0125")
    OPENAI_EVALUATION_MODEL = os.getenv("OPENAI_EVALUATION_MODEL", "gpt-3.5-turbo-0125")
    # if set, additional instructions will be cleared between tasks
    CLEAR_ADDITIONAL_INSTRUCTIONS_BETWEEN_TASKS = bool(
        int(os.getenv("CLEAR_ADDITIONAL_INSTRUCTIONS_BETWEEN_TASKS", 1))
    )
    PROMPT_TEMPLATE = os.getenv(
        "PROMPT_TEMPLATE",
        '**Question to answer**:\n\n"{text}"\n\n**Additional instructions (may be empty)**:\n\n"{prompt}"',
    )
    SUPPORTED_INPUTS = ("Text",)

    def setup(self):
        pass

    def _get_text(self, task_data, object_tag):
        """ """
        data = task_data.get(object_tag.value_name)
        return data

    def _get_prompts(self, context, prompt_tag) -> List[str]:
        """Getting prompt values"""
        if context:
            # Interactive mode - get prompt from context
            result = context.get("result")
            for item in result:
                if item.get("from_name") == prompt_tag.name:
                    return item["value"]["text"]
        # Initializing - get existing prompt from storage
        elif prompt := self.get(prompt_tag.name):
            return [prompt]

        return []

    def _find_textarea_tag(self, prompt_tag, object_tag):
        """Free-form text predictor"""
        li = self.label_interface

        try:
            textarea_from_name, _, _ = li.get_first_tag_occurence(
                "TextArea",
                self.SUPPORTED_INPUTS,
                name_filter=lambda s: s != prompt_tag.name,
                to_name_filter=lambda s: s == object_tag.name,
            )

            return li.get_control(textarea_from_name)
        except:
            return None

    def _find_tags_with_prefix(self, prefix, tag_type) -> Tuple[ControlTag, ObjectTag]:
        """Find tags in the config matching prefix and tag type"""
        li = self.label_interface
        tag_from_name, tag_to_name, value = li.get_first_tag_occurence(
            tag_type,
            # supported input types
            self.SUPPORTED_INPUTS,
            # if multiple <TextArea> are presented, use one with prefix specified
            name_filter=lambda s: s.startswith(prefix),
        )

        return li.get_control(tag_from_name), li.get_object(tag_to_name)

    def _validate_tags(self, textarea_tag: str) -> None:
        if not textarea_tag:
            raise ValueError("No supported tags found: <TextArea>")

    def _generate_normalized_prompt(
        self, text: str, prompt: str, task_data: Dict
    ) -> str:
        """ """
        return self.PROMPT_TEMPLATE.format(text=text, prompt=prompt)

    def _generate_response_regions(
        self,
        answer: str,
        ragas_eval: List[str],
        float_eval: Optional[float],
        str_eval: str,
        documentation: List[str],
        prompt_tag,
        textarea_tag: ControlTag,
        prompts: List[str],
    ) -> List:
        """ """
        regions = []

        if textarea_tag:
            regions.append(textarea_tag.label(text=[answer]))

        # not sure why we need this but it was in the original code
        regions.append(prompt_tag.label(text=prompts))

        doc_tag, _ = self._find_tags_with_prefix("documentation", "TextArea")
        eval_tag, _ = self._find_tags_with_prefix("evaluation", "TextArea")
        ragas_tag, _ = self._find_tags_with_prefix("ragas", "TextArea")
        eval_number_tag, _ = self._find_tags_with_prefix("float_eval", "Number")

        regions.append(doc_tag.label(text=documentation))
        regions.append(eval_tag.label(text=[str_eval]))
        regions.append(ragas_tag.label(text=ragas_eval))
        regions.append(eval_number_tag.label(number=int(float_eval * 100)))

        return regions

    def _predict_single_task(
        self,
        task_data: Dict,
        prompt_tag: Any,
        object_tag: Any,
        prompt: str,
        textarea_tag: ControlTag,
        prompts: List[str],
    ) -> Dict:
        """ """
        text = self._get_text(task_data, object_tag)
        norm_prompt = self._generate_normalized_prompt(text, prompt, task_data)
        response = get_rag_response(norm_prompt, self.extra_params)
        textual_evaluation = get_textual_evaluation(response)
        ragas_evaluation, ragas_evaluation_float = get_ragas_evaluation(response)

        regions = self._generate_response_regions(
            response["answer"],
            ragas_evaluation,
            ragas_evaluation_float,
            textual_evaluation,
            format_documentation(response["context"]),
            prompt_tag,
            textarea_tag,
            prompts,
        )

        return PredictionValue(
            result=regions,
            score=ragas_evaluation_float,
            model_version=str(self.model_version),
        )

    def predict(
        self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs
    ) -> ModelResponse:
        """ """
        global last_prompts
        predictions = []

        # prompt tag contains the prompt in the config
        # object tag contains what we plan to label
        prompt_tag, object_tag = self._find_tags_with_prefix("prompt", "TextArea")
        prompts = self._get_prompts(context, prompt_tag)

        if self.CLEAR_ADDITIONAL_INSTRUCTIONS_BETWEEN_TASKS:
            # hack: let's say if there are no predictions, then there are no prompts
            if not tasks[0].get("predictions"):
                prompts = []
        else:
            if not tasks[0].get("predictions") and last_prompts:
                prompts = last_prompts
        last_prompts = prompts

        prompt = "\n".join(prompts)

        textarea_tag = self._find_textarea_tag(prompt_tag, object_tag)
        self._validate_tags(textarea_tag)

        for task in tasks:
            # preload all task data fields, they are needed for prompt
            task_data = self.preload_task_data(task, task["data"])
            pred = self._predict_single_task(
                task_data, prompt_tag, object_tag, prompt, textarea_tag, prompts
            )
            predictions.append(pred)

        return ModelResponse(predictions=predictions)
