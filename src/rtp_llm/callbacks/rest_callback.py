from .base import BaseCallback, ResponseTransformation
from typing import Optional
import logging
import httpx
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)




class ResponseTransformationRequest(BaseModel):
    text: Optional[str] = None
    post_action_endpoint: Optional[str] = None # must be endpoint that returns nothing and uses post method that accepts uid as a parameter

class RestCallback(BaseCallback):
    """
    Callback that sends events to a REST API via POST requests.
    NOTE: all endpoints are optional, if not provided, the event will not be sent.
    """    
    def __init__(self, 
                 base_url: str,
                 on_response_endpoint: Optional[str] = None,
                 on_start_endpoint: Optional[str] = None,
                 on_error_endpoint: Optional[str] = None,
                 on_finish_endpoint: Optional[str] = None,
                 **kwargs):
        
        self.base_url = base_url
        self.on_response_endpoint = on_response_endpoint
        self.on_start_endpoint = on_start_endpoint
        self.on_error_endpoint = on_error_endpoint
        self.on_finish_endpoint = on_finish_endpoint

        if not any([self.on_response_endpoint, 
            self.on_start_endpoint, 
            self.on_error_endpoint, 
            self.on_finish_endpoint]):
            logger.warning("No endpoints were provided, this callback is useless")

        self.client = httpx.AsyncClient(base_url=base_url, **kwargs)
        
    def __create_data(self, uid: str, text: str = None) -> dict:
        data = {"uid": uid}
        if text is not None:
            data["text"] = text
        return data

    def __create_error_data(self, uid: str, error: Exception) -> dict:
        return {
            "uid": uid,
            "error": {
                "type": type(error).__name__,
                "message": str(error)
            }
        }

    async def __make_request(self, endpoint: str, data: dict) -> None:
        """Make an HTTP POST request and handle errors gracefully"""
        try:
            if self.client is None:
                logger.warning("Client is closed, cannot make request")
                return
            async with self.client as client:
                response = await client.post(endpoint, json=data)
            response.raise_for_status()
            logger.info(f"Successfully sent callback to {endpoint}")
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"HTTP error when calling {endpoint}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error when calling {endpoint}: {e}")

    async def on_response(self, uid: str, text: str) -> Optional[ResponseTransformation]:
        if self.on_response_endpoint:
            data = self.__create_data(uid, text)
            response = await self.__make_request(self.on_response_endpoint, data)
            try:
                response_transformation = ResponseTransformationRequest.model_validate(response)
                text = response_transformation.text
                # initing coroutin
                post_action = self.__make_request(
                    response_transformation.post_action_endpoint, 
                    self.__create_data(uid)
                    )
            except ValidationError as e:
                logger.error(f"Invalid response from {self.on_response_endpoint}: {e}")
                return None
            return ResponseTransformation(text=text, post_action=post_action)
        return None

    async def on_start(self, uid: str):
        if self.on_start_endpoint:
            data = self.__create_data(uid)
            await self.__make_request(self.on_start_endpoint, data)
    
    async def on_error(self, uid: str, error: Exception):
        if self.on_error_endpoint:
            data = self.__create_error_data(uid, error)
            await self.__make_request(self.on_error_endpoint, data)

    async def on_finish(self, uid: str):
        if self.on_finish_endpoint:
            data = self.__create_data(uid)
            await self.__make_request(self.on_finish_endpoint, data)
        self.client = None