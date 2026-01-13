import torch

from abc import ABC, abstractmethod
from shutil import get_terminal_size

from src.utils.checkpoint import load_model
from src.utils.device import get_device
from src.utils.logger import g_logger
from src.utils.tokenization import text_to_token_ids


class InteractiveSession(ABC):

    def __init__(self, model_path: str, max_new_tokens: int = None, temperature: float = 0.0,
                 top_k: int = None, device_type: str = "auto", seed: int = 123) -> None:
        self.model = None
        self._model_path = model_path
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._top_k = top_k
        self._device = get_device(device_type)
        self._seed = seed

    def load(self) -> None:
        self.model = load_model(self._model_path, self._device)[0]
        self.model.eval()

    @property
    @abstractmethod
    def welcome_msg(self) -> str:
        pass

    @abstractmethod
    def format_prompt(self, user_input: str) -> str:
        pass

    def start(self) -> None:
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
                idx = text_to_token_ids(prompt).to(self._device)
                self.model.generate(idx, self._max_new_tokens, self.model.config.context_length, self._temperature,
                                    self._top_k, live=True)
                print()  # Newline after generation ends

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break

            except Exception as e:
                g_logger.error(f"An error occurred: {e}")
