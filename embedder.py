from typing import List, Tuple, Dict, Any, Optional
import torch

from langchain_core.embeddings import Embeddings
from transformers import AutoTokenizer
from adapters import AutoAdapterModel


class SPECTEREmbeddings(Embeddings):
    def __init__(self, device: str = "cuda:0", **kwargs: Any):
        super().__init__(**kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        self._model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
        self._device = device

    @classmethod
    def class_name(cls) -> str:
        return "specter"

    def _embed_inputs(self, batch: List[str]) -> List[float]:
        inputs = self._tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
            max_length=512,
        ).to(self._device)

        self._model = self._model.to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        embeddings = outputs.last_hidden_state[:, 0, :]

        if embeddings.is_cuda:
            embeddings = embeddings.cpu()

        return (
            embeddings.numpy().flatten().tolist()
            if embeddings.shape[0] < 2
            else embeddings.numpy().tolist()
        )

    def embed_documents(self, texts: List[str]):
        self._model.load_adapter(
            "allenai/specter2",
            source="hf",
            load_as="specter2_proximity",
            set_active=True,
        )
        return self._embed_inputs(texts)

    def embed_query(self, text: str):
        self._model.load_adapter(
            "allenai/specter2_adhoc_query",
            source="hf",
            load_as="specter2_adhoc_query",
            set_active=True,
        )
        return self._embed_inputs([text])
