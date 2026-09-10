"""Small wire-compatible subset of the Isaac-GR00T ZMQ PolicyServer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import functools
import io
from typing import Any, Callable

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq


@dataclass(frozen=True)
class WireModalityConfig:
    """Lightweight counterpart serialized as GR00T's ``ModalityConfig``."""

    delta_indices: list[int]
    modality_keys: list[str]
    action_configs: list[dict[str, Any]] | None = None


class MsgSerializer:
    """Encode the same NumPy/msgpack envelopes as GR00T's PolicyClient."""

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        default = functools.partial(MsgSerializer._safe_encode, chain=None)
        return msgpack.packb(data, default=default)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        hook = functools.partial(MsgSerializer._safe_decode, chain=None)
        return msgpack.unpackb(data, object_hook=hook, raw=False)

    @staticmethod
    def _safe_encode(obj: Any, chain: Callable | None = None) -> Any:
        del chain
        if isinstance(obj, WireModalityConfig):
            return {"__ModalityConfig__": True, "as_json": asdict(obj)}
        if isinstance(obj, np.ndarray) and obj.dtype.kind == "O":
            raise TypeError("refusing to encode an object-dtype ndarray")
        return mnp.encode(obj)

    @staticmethod
    def _safe_decode(obj: Any, chain: Callable | None = None) -> Any:
        del chain
        if isinstance(obj, dict):
            marker = obj.get("__ndarray_class__", obj.get(b"__ndarray_class__"))
            if marker:
                payload = obj.get("as_npy", obj.get(b"as_npy"))
                if payload is None:
                    raise ValueError("malformed ndarray payload")
                return np.load(io.BytesIO(payload), allow_pickle=False)
            nd_value = obj.get(b"nd", obj.get("nd"))
            kind = obj.get(b"kind", obj.get("kind"))
            if nd_value and kind in (b"O", "O"):
                raise ValueError("refusing to decode an object-dtype ndarray payload")
        return mnp.decode(obj)


@dataclass(frozen=True)
class _Endpoint:
    function: Callable[..., Any]
    requires_input: bool = True


class PolicyServer:
    """Serve ``get_action`` using the protocol expected by GR00T PolicyClient."""

    def __init__(
        self,
        policy: Any,
        *,
        host: str = "*",
        port: int = 5550,
        api_token: str | None = None,
    ) -> None:
        self.policy = policy
        self.host = host
        self.port = port
        self.api_token = api_token
        self.running = True
        self._closed = False
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self._endpoints = {
            "ping": _Endpoint(lambda: {"status": "ok", "message": "Server is running"}, False),
            "kill": _Endpoint(self._kill, False),
            "get_action": _Endpoint(policy.get_action),
            "reset": _Endpoint(policy.reset),
            "get_modality_config": _Endpoint(policy.get_modality_config, False),
        }

    def _kill(self) -> None:
        self.running = False

    def run(self) -> None:
        address = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"Server is ready and listening on {address}")
        while self.running:
            try:
                request = MsgSerializer.from_bytes(self.socket.recv())
                if self.api_token is not None and request.get("api_token") != self.api_token:
                    self.socket.send(MsgSerializer.to_bytes({"error": "Unauthorized"}))
                    continue
                endpoint_name = request.get("endpoint", "get_action")
                if endpoint_name not in self._endpoints:
                    raise ValueError(f"unknown endpoint: {endpoint_name}")
                endpoint = self._endpoints[endpoint_name]
                result = (
                    endpoint.function(**request.get("data", {}))
                    if endpoint.requires_input
                    else endpoint.function()
                )
                self.socket.send(MsgSerializer.to_bytes(result))
            except Exception as error:
                import traceback

                traceback.print_exc()
                self.socket.send(MsgSerializer.to_bytes({"error": str(error)}))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.running = False
        self.socket.close(linger=0)
        self.context.term()

    def __enter__(self) -> "PolicyServer":
        print(f"\nACT-Sonic PolicyServer listening on {self.host}:{self.port}\n")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
