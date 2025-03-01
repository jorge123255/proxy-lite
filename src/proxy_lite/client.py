import os
from abc import ABC, abstractmethod
from functools import cached_property
from typing import ClassVar, Literal, Optional, Union

import httpx
from httpx import Limits, Timeout
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import (
    ChatCompletion,
)
from pydantic import BaseModel

from proxy_lite.history import MessageHistory
from proxy_lite.logger import logger
from proxy_lite.serializer import (
    BaseSerializer,
    OpenAICompatibleSerializer,
)
from proxy_lite.tools import Tool


class BaseClientConfig(BaseModel):
    http_timeout: float = 50
    http_concurrent_connections: int = 50


class BaseClient(BaseModel, ABC):
    config: BaseClientConfig
    serializer: ClassVar[BaseSerializer]

    @abstractmethod
    async def create_completion(
        self,
        messages: MessageHistory,
        temperature: float = 0.7,
        seed: Optional[int] = None,
        tools: Optional[list[Tool]] = None,
        response_format: Optional[type[BaseModel]] = None,
    ) -> ChatCompletion: ...

    """
    Create completion from model.
    Expect subclasses to adapt from various endpoints that will handle
    requests differently, make sure to raise appropriate warnings.

    Returns:
        ChatCompletion: OpenAI ChatCompletion format for consistency
    """

    @classmethod
    def create(cls, config: BaseClientConfig) -> "BaseClient":
        supported_clients = {
            "openai-azure": OpenAIClient,
            "convergence": ConvergenceClient,
        }
        if config.name not in supported_clients:
            error_message = f"Unsupported model: {config.name}."
            raise ValueError(error_message)
        return supported_clients[config.name](config=config)

    @property
    def http_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=Timeout(self.config.http_timeout),
            limits=Limits(
                max_connections=self.config.http_concurrent_connections,
                max_keepalive_connections=self.config.http_concurrent_connections,
            ),
        )


class OpenAIClientConfig(BaseClientConfig):
    name: Literal["openai"] = "openai"
    model_id: str = "gpt-4o"
    api_key: str = os.environ.get("OPENAI_API_KEY")


class OpenAIClient(BaseClient):
    config: OpenAIClientConfig
    serializer: ClassVar[OpenAICompatibleSerializer] = OpenAICompatibleSerializer()

    @cached_property
    def external_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=self.config.api_key,
            http_client=self.http_client,
        )

    async def create_completion(
        self,
        messages: MessageHistory,
        temperature: float = 0.7,
        seed: Optional[int] = None,
        tools: Optional[list[Tool]] = None,
        response_format: Optional[type[BaseModel]] = None,
    ) -> ChatCompletion:
        base_params = {
            "model": self.config.model_id,
            "messages": self.serializer.serialize_messages(messages),
            "temperature": temperature,
        }
        optional_params = {
            "seed": seed,
            "tools": self.serializer.serialize_tools(tools) if tools else None,
            "tool_choice": "required" if tools else None,
            "response_format": {"type": "json_object"} if response_format else {"type": "text"},
        }
        base_params.update({k: v for k, v in optional_params.items() if v is not None})
        return await self.external_client.chat.completions.create(**base_params)


class ConvergenceClientConfig(BaseClientConfig):
    name: Literal["convergence"] = "convergence"
    model_id: str = "convergence-ai/proxy-lite-7b"
    api_base: str = "http://localhost:8000/v1"
    api_key: str = "none"


class ConvergenceClient(OpenAIClient):
    config: ConvergenceClientConfig
    serializer: ClassVar[OpenAICompatibleSerializer] = OpenAICompatibleSerializer()
    _model_validated: bool = False

    async def _validate_model(self) -> None:
        try:
            response = await self.external_client.models.list()
            assert self.config.model_id in [model.id for model in response.data], (
                f"Model {self.config.model_id} not found in {response.data}"
            )
            self._model_validated = True
            logger.debug(f"Model {self.config.model_id} validated and connected to cluster")
        except Exception as e:
            logger.error(f"Error retrieving model: {e}")
            raise e

    @cached_property
    def external_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.api_base,
            http_client=self.http_client,
        )

    async def create_completion(
        self,
        messages: MessageHistory,
        temperature: float = 0.7,
        seed: Optional[int] = None,
        tools: Optional[list[Tool]] = None,
        response_format: Optional[type[BaseModel]] = None,
    ) -> ChatCompletion:
        if not self._model_validated:
            await self._validate_model()
        base_params = {
            "model": self.config.model_id,
            "messages": self.serializer.serialize_messages(messages),
            "temperature": temperature,
        }
        optional_params = {
            "seed": seed,
            "tools": self.serializer.serialize_tools(tools) if tools else None,
            "tool_choice": "auto" if tools else None,  # vLLM does not support "required"
            "response_format": response_format if response_format else {"type": "text"},
        }
        base_params.update({k: v for k, v in optional_params.items() if v is not None})
        return await self.external_client.chat.completions.create(**base_params)

    async def unload_model(self) -> None:
        """Unload the model from GPU memory"""
        try:
            await self.external_client.delete(f"/models/{self.config.model_id}")
            self._model_validated = False
            logger.debug(f"Model {self.config.model_id} unloaded from GPU")
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
            raise e


ClientConfigTypes = Union[OpenAIClientConfig, ConvergenceClientConfig]
ClientTypes = Union[OpenAIClient, ConvergenceClient]
