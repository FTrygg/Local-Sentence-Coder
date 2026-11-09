import subprocess
import json
import time
import requests
import logging
from importlib.resources import files

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)  # Set the logger to capture warnings and above

class LocalSentenceCoder:
    def __init__(self, model_name: str = "qwen3:14b", host : str = "http://localhost:11434") -> None:
        self.host = host
        self.model_name = model_name

        logger.info(f"Initializing LocalSentenceCoder with model '{self.model_name}' at host '{self.host}'.")
        self.load_examples()
        self.prompt = self.__load_prompt__()
        self.LLM_return_schema = self.create_schema()


    def is_ollama_running(self, timeout : int = 20 ) -> bool:
        """Checks if the Ollama server is running by sending a GET request to the server."""
        try:
            response = requests.get(f"{self.host}/", timeout=timeout)
            return response.status_code == 200
        
        except requests.ConnectionError:
            logger.error("Failed to connect to Ollama server.")
            return False

    def start_ollama(self) -> None:
        """Starts the Ollama background service on Windows."""
        logger.info("Ollama is not running. Attempting to start it...")

        try:
            subprocess.Popen(
                ["ollama", "serve"], 
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                "Could not find the 'ollama' executable. Ensure Ollama is installed and added to your system PATH."
            )

        # Wait for the server to spin up
        for _ in range(20):  # Give it up to 20 seconds
            if self.is_ollama_running():
                logger.info("Ollama started successfully.")
                return
            time.sleep(1)
        
        raise TimeoutError("Timed out waiting for Ollama server to start.")

    def ensure_model_loaded(self) -> None:
        """Ensures the specific model is loaded into memory or ready."""
        logger.info(f"Ensuring model '{self.model_name}' is available and loaded...")
    
        try:
            # Sends a lightweight warmup request to force Ollama to load the model into VRAM/RAM
            response = requests.post(
                f"{self.host}/api/generate",
                json={"model": self.model_name, "prompt": "", "keep_alive": "10m"},
                timeout=60  # Larger models like 14B can take a moment to load into memory
            )
            if response.status_code != 200:
                logger.error(f"Error loading model. Response: {response.text}")
                logger.error("Tip: Make sure you have downloaded the model by running 'ollama run qwen3:14b' or 'ollama pull qwen3:14b' in your terminal.")
            else:
                logger.info(f"Model '{self.model_name}' is loaded and ready.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")


    def __load_prompt__(self) -> str:
       """Loads the prompt from the 'Prompt.txt' file."""
       prompt = files("Coder").joinpath("Prompt.txt").read_text(encoding="utf-8")
       logger.debug(f"Prompt: \n{prompt}")
       return prompt
    
    def load_examples(self) -> None:
        """Loads examples from the 'examples.json' file """
        json_path = files("Coder").joinpath("Coding_Examples.json")
        logger.debug(f"Loading examples from: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.code_scheme = data["codescheme"]
            examples = data["examples"]
            self.__examples_input__ = {i:{"sentence": s["sentence"]} for i, s in enumerate(examples)}
            self.__examples_output__ = {"results" : [{"id": i, self.code_scheme: s["flag"]} for i, s in enumerate(examples)]}
            
    def create_schema(self) -> dict:
        """Creates a JSON schema for the expected output."""
        schema = {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            self.code_scheme: {"type": "boolean"},
                        },
                        "required": ["id", self.code_scheme],
                    },
                }
            },
            "required": ["results"],
        }
        logger.debug(f"Created JSON schema: {json.dumps(schema, indent=2)}")
        return schema

    def __sentences_to_dict__(self, sentences: list[str]) -> dict[int, dict[str, str]]:
        """Converts a list of sentences into a JSON structure."""
        sentence_dict = {i:{"sentence": s} for i, s in enumerate(sentences)}
        return sentence_dict
    
    def create_prompt(self, sentences) -> str:
        """Creates a prompt for the model based on the input text."""

        return f"""
            {self.prompt}
            Example input: {self.__examples_input__}
            Example output: {self.__examples_output__}
            Sentences: {sentences}"""

    def poll_model(self, prompt: str, timeout: int = 30) -> requests.Response:
        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "format": self.LLM_return_schema,
                "think": False,
                "options": {"temperature": 0, "seed": 1, "top_k": 1},
            },
            timeout=300,
        )
        response.raise_for_status()
        return response

    def code_sentences(self, sentences: list[str], batch_size: int = 20, pad_before: int = 3, NUM_CTX : int = 8192) -> dict:
        """Codes the given sentences using the model."""
        if not self.is_ollama_running():
            self.start_ollama()
        
        self.ensure_model_loaded()

        sentence_dict = self.__sentences_to_dict__(sentences)

        for start in range(0, len(sentences), batch_size):
            end = min(start + batch_size, len(sentences))
            # padded window: only look backwards, clamped at the document start
            ctx_start = max(0, start - pad_before)

            subset = {k: sentence_dict[k] for k in range(ctx_start, end)}
            prompt = self.create_prompt(subset)
 

            res = self.poll_model(prompt)
            body = res.json()

            # 1) did generation finish normally?
            if not body.get("done") or body.get("done_reason") != "stop":
                logging.warning(f"Generation did not finish cleanly!")
            
            # 2) was the prompt silently truncated?
            prompt_tokens = body.get("prompt_eval_count", 0)
            if prompt_tokens >= NUM_CTX:
                logging.warning(f"Prompt used {prompt_tokens} tokens, context limit of {NUM_CTX} reached.")
#
            flagged_sentences = json.loads(body["response"])["results"]
            #now we check whether all ids are present in the results, if not we have a problem
            ids = [r["id"] for r in flagged_sentences]
            expected_ids = list(subset.keys())
            if set(ids) != set(expected_ids):
                logging.warning(f"Mismatch in IDs. Expected {expected_ids}, got {ids}.")
            
            for sentence in flagged_sentences:
                id_ = sentence.get("id")
                is_flagged = sentence.get(self.code_scheme)
                if not self.code_scheme in sentence_dict[id_].keys():
                    sentence_dict[id_][self.code_scheme] = is_flagged

        return sentence_dict