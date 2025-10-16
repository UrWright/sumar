import logging, logging.config, yaml
import requests
from typing import Dict, Optional, List, Literal
from split_text import TextSplitProcess
from dataclasses import dataclass
from dataclasses_json import dataclass_json

# @dataclass_json
# @dataclass
# class AppResponse:
#    session_id: str
#    result: Union[str, Dict, List]


# @dataclass_json
# @dataclass
# class AppError:
#    session_id: str
#    error: str
#    notes: str


class AppConfig:
    def __init__(self, logger_name="sumar", model_opt=None):
        """
        logger_name: name of log provider in log.yaml
        model_opt: (llm provider:index) the model index under llm provider in llm.yaml
        """
        self.model_provider = None
        self.logger = None
        self.chunkiter = None
        self._model_specs = None
        self.setup_logger_provider(logger_name)

        if model_opt:
            self.setup_model_provider(model_opt)

    def setup_model_provider(self, model_opt):
        try:
            with open("configs/llm.yaml", "r", encoding="utf-8") as f:
                config: Dict = yaml.safe_load(f)
                llm_model_opt = model_opt.split(":")
                model_provider: Dict = config.get(llm_model_opt[0])
                model_provider["model"] = model_provider.get("model")[int(llm_model_opt[1])]
                self.model_provider = model_provider
        except Exception as e:
            raise Exception(f"WARNING: load configs/llm.yaml files: {e}")

    def setup_logger_provider(self, logger_name):
        if self.logger:
            return self.logger
        try:
            with open("configs/log.yaml", "r", encoding="utf-8") as f:
                config: Dict = yaml.safe_load(f)
                logging.config.dictConfig(config)
                self.logger = logging.getLogger(logger_name)
        except Exception as e:
            print(f"WARNING: load configs/log.yaml files: {e}")
            self.logger = logging.getLogger("sumar")
            self.logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        return self.logger

    async def setup_text_splitter(self):
        if self.chunkiter:
            return self.chunkiter
        try:
            with open("configs/seg.yaml", "r", encoding="utf-8") as f:
                config: Dict = yaml.safe_load(f)
                splitter: Dict = config.get("sat")
                tiktoken: Dict = config.get("tiktoken")
                self.chunkiter = TextSplitProcess(
                    self.logger,
                    splitter_model_dir=splitter.get("model_dir"),
                    spliter_model_repo=splitter.get("model_repo"),
                    spliter_tokenizer_repo=splitter.get("tokenizer_repo"),
                    tiktoken_model_dir=tiktoken.get("model_dir"),
                    tiktoken_model_repo=tiktoken.get("model_repo"),
                )
            await self.chunkiter.load_models()
            return self.chunkiter
        except Exception as e:
            raise Exception(f"WARNING: load text splitter : {e}")

    @property
    def model_specs(self):
        if self._model_specs is not None:
            return self._model_specs

        self.logger.info(f"model provider: {self.model_provider}")
        model_info_url = f'{self.model_provider.get("base_url")}/models'
        model_name = self.model_provider.get("model")

        try:
            result = requests.get(model_info_url, timeout=5)
            data: List[Dict] = result.json().get("data", [])
            if not data:
                raise ValueError("data not found")

            for d in data:
                if d.get("id", "") == model_name:
                    self._model_specs = d
                    return self._model_specs
            raise ValueError(f"model '{model_name}' not found")

        except Exception as e:
            self.logger.warning(f"failed to get model specs from '{model_info_url}': {e}")
            self._model_specs = None

        return self._model_specs
