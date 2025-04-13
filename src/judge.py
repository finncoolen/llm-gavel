"""LLM as Judge metrics based on user defined prompts."""

from abc import abstractmethod
from dataclasses import dataclass, field

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BatchEncoding


@dataclass(frozen=True)
class Prompt:
    """A data class for managing and formatting prompts used in LLM as Judge."""

    name: str
    prompt_text: str
    replacement_fields: list[str] = field(default_factory=list)
    output_categories: list[str] = field(default_factory=list)


class AbstractLLMAsJudgeMetric:
    """Base metric subclass class for LLMAsJudge."""

    @abstractmethod
    def compute(
        self, inputs: list[str] | list[list[str]], prompt: Prompt
    ) -> dict[str, list[float]]:
        """Compute a custom LLMAsJudge metric for output text."""


class LLMAsJudge(AbstractLLMAsJudgeMetric):
    """A class that uses a language model as a judge to compute outputs."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cpu",
        cache_dir: str | None = None,
        max_length: int = 1024,
    ) -> None:
        """
        Initialize the LLMAsJudge with a model specified by the path or identifier.

        Parameters
        ----------
        model_name_or_path : str
            The path or identifier of the pretrained model.
        device : str, optional
            The device to run the model on. Defaults to "cpu".
        cache_dir : str, optional
            Directory for caching model and tokenizer data.
        max_length : int, optional
            The maximum sequence length that the model can accept. Defaults to 1024.

        """
        self.device = device
        self.max_length = max_length
        self.config = AutoConfig.from_pretrained(
            model_name_or_path, cache_dir=cache_dir
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, cache_dir=cache_dir
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=self.config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        self.model.eval()
        self.model_name = model_name_or_path
        self.model.to(device)
        self.softmax = nn.Softmax(dim=1)

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.model.resize_token_embeddings(len(self.tokenizer))

    def format_prompt(self, input_text: list[str], prompt: Prompt) -> str:
        """
        Prepare and format input prompt using huggingface chat template.

        Parameters
        ----------
        input_text : list[str]
            A list of input_text, each being a string or list of strings.
        prompt : Prompt
            A prompt object specifying text formatting and placeholders.

        Returns
        -------
        str
            formatted input string

        """
        messages = [
            {
                "role": "user",
                "content": prompt.prompt_text.format(
                    **dict(zip(prompt.replacement_fields, input_text, strict=True))
                ),
            }
        ]

        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    def _call_model(self, encoded_list: BatchEncoding) -> torch.Tensor:
        """
        Call the model with encoded inputs and return softmax probabilities.

        Parameters
        ----------
        encoded_list : transformers.BatchEncoding
            The batch encoding of input text.

        Returns
        -------
        torch.Tensor
            Softmax probabilities for the next tokens in the input sequence.

        """
        output = self.model(
            input_ids=encoded_list["input_ids"],
            attention_mask=encoded_list["attention_mask"],
        )
        logits = output.logits
        next_token_logits = logits[:, -1, :]
        return self.softmax(next_token_logits)

    def generate_token_permutations(
        self, formatted_prompt: str, category_tokens: list[list[int]]
    ) -> list[list[int]]:
        """
        Generate all possible permutations of token sequences for each output category.

        Parameters
        ----------
        formatted_prompt : str
            The formatted input prompt.
        prompt : Prompt
            A prompt object specifying text formatting and output categories.
        category_tokens : list[list[int]]
            Encoded tokens for each of the output category strings

        Returns
        -------
        list[list[int]]
            list of token sequences for each multi token category

        """
        # tokenized encoding of prompt to be passed to model
        encoded_prompt = self.tokenizer(
            formatted_prompt,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=False,
            padding=False,
            return_tensors="pt",
        ).to(self.device)

        cumulative_lists = encoded_prompt.input_ids.tolist()

        for category_token_list in category_tokens:
            # Start with a fresh copy of the encoded prompt for each category
            current_list = encoded_prompt.input_ids.tolist()[0].copy()

            for token in category_token_list[:-1]:
                # for each token in the category, append it to the list
                # - not including the last token (we dont need to predict beyond)
                current_list.append(token)
                cumulative_lists.append(current_list.copy())  # Append a copy of the
                # list including the next token

        return cumulative_lists

    def calculate_probabilities(
        self,
        category_tokens: list[list[int]],
        batch_probabilities: torch.Tensor,
        prompt: Prompt,
    ) -> dict[str, float]:
        """
        Calculate the probability of each output category.

        Parameters
        ----------
        category_tokens : list[list[int]]
            A list of lists of tokens for each output category.
        batch_probabilities : torch.Tensor
            Softmax probabilities for the next tokens in the input sequence.
        prompt: Prompt
            Prompt object used

        Returns
        -------
        dict[str,float]
            A dictionary containing the probability of each output category.

        """
        results_dict = {}
        for i, token_list in enumerate(category_tokens):
            probability = 1.0
            for j in token_list:
                if token_list.index(j) == 0:
                    probability *= batch_probabilities[0][j]

                else:
                    probability *= batch_probabilities.pop(1)[j]

            results_dict[prompt.output_categories[i]] = probability

        return results_dict

    def compute(self, input_text: list[str], prompt: Prompt) -> dict[str, list[float]]:
        """
        Compute model output probability for each output category token.

        Parameters
        ----------
        input_text : list[str]
            Inputs to process, each as a string or list of strings.
        prompt : Prompt
            The prompt object guiding input formatting and expected outputs.

        Returns
        -------
        dict[str, list[float]]
            A dictionary containing categorized results as specified
            by the prompt's output categories.

        """
        formatted_prompt = self.format_prompt(input_text, prompt)

        # list of lists of tokens - for each of the output categories
        category_tokens = self.tokenizer(
            prompt.output_categories, add_special_tokens=False
        ).input_ids

        cumulative_lists = self.generate_token_permutations(
            formatted_prompt, category_tokens
        )

        batch_encoded_prompts = self.tokenizer(
            self.tokenizer.batch_decode(cumulative_lists, skip_special_tokens=True),
            add_special_tokens=False,
            padding=True,
            truncation=False,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            batch_probabilities = self._call_model(batch_encoded_prompts).tolist()

        return self.calculate_probabilities(
            category_tokens, batch_probabilities, prompt
        )
