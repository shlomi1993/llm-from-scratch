import torch

from abc import ABC, abstractmethod
from shutil import get_terminal_size

from src.utils.checkpoint import load_model
from src.utils.device import get_device
from src.utils.logger import g_logger
from src.utils.tokenization import g_tokenizer


class InteractiveSession(ABC):
    """
    Abstract base class for interactive sessions with language models.
    """

    def __init__(self, model_path: str, max_new_tokens: int = None, temperature: float = 0.0,
                 top_k: int = None, device_type: str = "auto", seed: int = 123) -> None:
        """
        Initialize the interactive session.

        Args:
            model_path (str): The path to the pre-trained language model.
            max_new_tokens (int): The maximum number of new tokens to generate per input. If None, uses model default.
            temperature (float): The sampling temperature for generation.
            top_k (int): The top-k sampling parameter. If None, uses model default.
            device_type (str): The device type to use ("auto", "cuda", "mps", or "cpu").
            seed (int): The random seed for reproducibility.
        """
        self.model = None
        self._model_path = model_path
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._top_k = top_k
        self._device = get_device(device_type)
        self._seed = seed

    def load(self) -> None:
        """
        Load the pre-trained model from the specified path and set it to eval mode.
        """
        self.model = load_model(self._model_path, self._device)[0]
        self.model.eval()

    @property
    @abstractmethod
    def welcome_msg(self) -> str:
        """
        Get the welcome message displayed at the start of the interactive session.
        """
        pass

    @abstractmethod
    def format_prompt(self, user_input: str) -> str:
        """
        Format the user input into a prompt suitable for the model.

        Args:
            user_input (str): The user's input.

        Returns:
            str: The formatted prompt.
        """
        pass

    def start(self) -> None:
        """
        Start the interactive session.

        Logic:
        - Load the model if not already loaded.
        - Display the welcome message.
        - Loop:
            - Get user input. If empty, continue. If input is "/bye", exit.
            - Format the input into a prompt.
            - Tokenize the prompt and move it to the appropriate device.
            - Generate a live response using the model (can be interrupted with KeyboardInterrupt).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() before starting the interactive session.")

        torch.manual_seed(self._seed)
        g_logger.info(f"Using device '{self._device.type}' and random seed {self._seed}")

        self.load()

        sep = "=" * get_terminal_size().columns
        print(f"{sep}\n{self.welcome_msg}\n{sep}")

        while True:
            try:
                user_input = input(">>> ").strip()
                if not user_input:
                    continue

                if user_input.lower() == "/bye":
                    print("Goodbye!")
                    break

                prompt = self.format_prompt(user_input)
                idx = g_tokenizer.text_to_token_ids(prompt).to(self._device)
                self.model.generate(idx, self._max_new_tokens, self.model.config.context_length, self._temperature,
                                    self._top_k, live=True)
                print()  # Newline after generation ends

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break

            except Exception as e:
                g_logger.error(f"An error occurred: {e}")
