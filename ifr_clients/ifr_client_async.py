import logging
from typing import Union, Dict, Any, List, Literal
import aiohttp
import msgpack
import ujson

from ifr_clients.common_utils import decode_face_data, b64_encode_data
from ifr_clients.schemas import RecognitionResponse

logger = logging.getLogger(__name__)


class IFRClientAsync:
    """
    Async client for interacting with InsightFace-REST API.

    Provides methods to query server information and perform face recognition/extraction.

    Attributes:
       server: Base URL of the InsightFace-REST server
       sess: Persistent HTTP session for connection pooling
    """

    def __init__(self, host: str = 'http://localhost', port: Union[str, int] = 18081) -> None:
        """
        Initialize the InsightFace-REST client.

        Args:
            host: Server hostname or IP address
            port: Server port number
        """
        self.server = f'{host}:{port}'
        self.sess: Union[aiohttp.ClientSession, None] = None

    async def start(self, sess: Union[aiohttp.ClientSession, None] = None):
        if sess:
            self.sess = sess
        else:
            self.sess = aiohttp.ClientSession()

    async def close(self):
        await self.sess.close()

    async def server_info(self, server: str = None, show: bool = True) -> Dict[str, Any]:

        """
        Get information about the IFR server.

        Args:
           server (str): URL of the IFR server. If not specified, uses the URL provided when the client was created.
           show (bool): Whether to print the server information.

        Returns:
           dict: Information about the IFR server.
        """

        if server is None:
            server = self.server

        info_uri = f'{server}/info'
        info = await self.sess.get(info_uri)
        info = await info.json()
        if show:
            # Extract relevant server information
            server_uri = self.server
            backend_name = info['models']['inference_backend']
            det_name = info['models']['det_name']
            rec_name = info['models']['rec_name']
            rec_batch_size = info['models']['rec_batch_size']
            det_batch_size = info['models']['det_batch_size']
            det_max_size = info['models']['max_size']

            # Format and display server information
            print(f'Server: {server_uri}\n'
                  f'    Inference backend:      {backend_name}\n'
                  f'    Detection model:        {det_name}\n'
                  f'    Detection image size:   {det_max_size}\n'
                  f'    Detection batch size:   {det_batch_size}\n'
                  f'    Recognition model:      {rec_name}\n'
                  f'    Recognition batch size: {rec_batch_size}')

        return info

    async def extract(
            self,
            data: List[Union[str, bytes]],
            mode: Literal['paths', 'data'] = 'paths',
            threshold: float = 0.6,
            extract_embedding: bool = True,
            return_face_data: bool = False,
            return_landmarks: bool = False,
            embed_only: bool = False,
            limit_faces: int = 0,
            min_face_size: int = 0,
            img_req_headers: Dict[str, str] = None,
            use_msgpack: bool = True,
            raw_response: bool = True
    ) -> Union[RecognitionResponse, dict]:

        """
        Perform face extraction and recognition on input data.

        Supports:
        - Image URI
        - Raw image bytes
        - Base64-encoded images

        Args:
            data: List of image paths (mode='paths') or image bytes (mode='data')
            mode: Input type - 'paths' for image URLs/paths, 'data' for binary images
            threshold: Confidence threshold for face detection (0.0-1.0)
            extract_embedding: Whether to calculate face embeddings
            return_face_data: Whether to include decoded face images in response
            return_landmarks: Whether to include facial landmarks in response
            embed_only: Set True if input images are pre-cropped faces (112x112)
            limit_faces: Maximum faces to process per image (0 = no limit)
            min_face_size: Faces smaller than this value will be ignored (0 = no limit)
            img_req_headers: Headers to use for requesting images from remote servers.
            use_msgpack: Use MessagePack for faster binary serialization and bandwidth savings.
            raw_response: Return raw dictionary instead of parsed response object

        Returns:
            Processed recognition results either as RecognitionResponse object or raw dict
        """

        extract_uri = f'{self.server}/extract'

        # Prepare image data based on input mode
        images: Dict[str, Any]
        if mode == 'data':
            if not use_msgpack:
                # Convert binary data to base64 strings if not using msgpack
                images = dict(data=b64_encode_data(data))
            else:
                images = dict(data=data)
        elif mode == 'paths':
            images = dict(urls=data)
        else:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'paths' or 'data'")

        req = dict(images=images,
                   threshold=threshold,
                   extract_ga=False,
                   extract_embedding=extract_embedding,
                   return_face_data=return_face_data,
                   return_landmarks=return_landmarks,
                   embed_only=embed_only,  # If set to true API expects each image to be 112x112 face crop
                   limit_faces=limit_faces,  # Limit maximum number of processed faces, 0 = no limit
                   min_face_size=min_face_size,
                   msgpack=use_msgpack,
                   img_req_headers=img_req_headers
                   )

        # Send request with appropriate serialization
        if use_msgpack:
            # MessagePack binary format
            resp = await self.sess.post(
                extract_uri,
                data=msgpack.dumps(req),
                timeout=120,
                headers={
                    'content-type': 'application/msgpack',
                    'accept': 'application/x-msgpack'
                }
            )
        else:
            # Standard JSON format
            resp = await self.sess.post(extract_uri, json=req, timeout=120)

        content = await resp.read()

        if resp.headers['content-type'] == 'application/x-msgpack':
            content = msgpack.loads(content)
        else:
            content = ujson.loads(content)

        # Decode face images if requested
        if return_face_data:
            content = decode_face_data(content)

        # Return either raw dict or validated Pydantic model
        return content if raw_response else RecognitionResponse.model_validate(content)
