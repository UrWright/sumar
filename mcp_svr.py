import json, re, uuid

from load_config import AppConfig
from split_text import TextSplitProcess, TextChunkItem

from contextlib import asynccontextmanager
from typing import Any, Dict, Union
import fastmcp

# from fastmcp.server.http import StarletteWithLifespan
# from fastmcp.server.dependencies import get_context

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.routing import APIRoute
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response, JSONResponse
from pydantic import BaseModel, ValidationError


class DataCommMessage(BaseModel):
    client: str = ""
    config: str = ""
    data: Any = None

    def parse_config(self, delimiter: str = ";", kv_separator: str = "=") -> Dict[str, Union[int, float, str]]:
        """
        Parse config string into a dictionary.
        Args:
            config: Config string, e.g. 'threshold=123;max_tokens=2048;note="hello; world"'
            delimiter: Segment delimiter, default ';'
            kv_separator: Key-value separator, default '='

        Returns:
            Dict with key-value pairs, int/float converted if possible
        """
        result = {}
        if not self.config:
            return result

        segments = []
        buf = ""
        in_quotes = False
        quote_char = ""

        for c in self.config:
            if c in ('"', "'"):
                if not in_quotes:
                    in_quotes = True
                    quote_char = c
                elif quote_char == c:
                    in_quotes = False
                buf += c
            elif c == delimiter and not in_quotes:
                if buf.strip():
                    segments.append(buf.strip())
                buf = ""
            else:
                buf += c
        if buf.strip():
            segments.append(buf.strip())

        for segment in segments:
            if kv_separator not in segment:
                continue
            key, value = segment.split(kv_separator, 1)
            key = key.strip()
            value = value.strip().strip("\"'")

            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass

            result[key] = value

        return result


class SessionIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        print(f"\n## MiddleWare.Request.Headers:: {request.headers}\n")
        response: Response = await call_next(request)
        return response


app = FastAPI(title="Text Segmentation")
app.add_middleware(SessionIDMiddleware)


@app.post("/v1/text/tokens", response_model=DataCommMessage)
async def calculate_text_in_tokens(request: Request, body: DataCommMessage):  #
    """
    Description:
        Calculate the token count of the given text for a specific session and
        return the result as a structured response message.

    Inputs:
        client (str): Unique identifier of the request session.
        config (str, optional): Additional configuration or metadata for the request.
        data (str): The text content to be analyzed and tokenized.

    Outputs:
        client (str): Same as the input session identifier.
        config (str): Indicates the response type — one of ['tokens', 'error', 'finish'].
        data (str | int): The corresponding result data — token count, error message, or completion notice.
    """
    try:
        # body: DataCommMessage = await request.body()
        print("\n\n HEADER: ", request.headers, "\n\n")
        print("\n\n BODY: ", body, "\n\n")
        data = body.data
        client = body.client

        if not data or not client:
            raise Exception("missing data / client ")

        chunkiter: TextSplitProcess = app.state.chunkiter
        app_config: AppConfig = app.state.app_config
        tokens: int = await chunkiter.count_tokens(data)
        return DataCommMessage(client=client, config="FINISH", data=tokens)

    except Exception as e:
        return DataCommMessage(client=client, config="error", data=f"{e}")


@app.post("/v1/text/segmentation")  # , response_model=DataCommMessage)
async def split_text_to_segments(request: Request, body: DataCommMessage):
    """
    Description:
        Split the input text into multiple segments for a given session, according to the specified token threshold.

    Inputs:
        client (str): Unique identifier of the request session.
        config (str): Segmentation configuration, specifying the token threshold (e.g., threshold=a number).
        data (str): The text content to be analyzed and segmented.

    Outputs:
        client (str): Matches the input session identifier.
        config (str): Indicates the response type — one of ['chunk=<size>', 'error', 'finish'].
        data (str): The segmented text content or corresponding error details.
    """

    async def text_to_chunk_json(client, text, threshold):
        chunkiter: TextSplitProcess = app.state.chunkiter
        app_config: AppConfig = app.state.app_config
        logger = app_config.logger

        try:
            text_tokens = await chunkiter.count_tokens(text)
            if text_tokens <= threshold:
                yield json.dumps(
                    DataCommMessage(
                        client=client,
                        config="FINISH",
                        data=TextChunkItem(tokens=text_tokens, text=text),
                    ).model_dump(),
                    ensure_ascii=False,
                ) + "\n"
                return

            sentences = await chunkiter.text_to_sentences(text)
            async for chunk in chunkiter.sentences_to_chunks(sentences, threshold):
                yield json.dumps(
                    DataCommMessage(client=client, config="FINISH", data=chunk).model_dump(),
                    ensure_ascii=False,
                ) + "\n"
        except Exception as e:
            yield json.dumps(DataCommMessage(client=client, config="error", data=f"{e}").model_dump()) + "\n"

    async def build_response_stream(client, config, data):
        yield json.dumps(
            DataCommMessage(
                client=client,
                config=config,
                data=f"{data}",
            ).model_dump(),
            ensure_ascii=False,
        ) + "\n"

    try:
        # body: DataCommMessage = await request.body()
        print("\n\n HEADER: ", request.headers, "\n\n")
        print("\n\n BODY: ", body, "\n\n")
        long_text = body.data
        client = body.client
        threshold = body.parse_config().get("threshold", 2048)

        if not long_text or not client:
            raise Exception("missing long text / session id")

        return StreamingResponse(
            text_to_chunk_json(client, long_text, threshold),
            # media_type="application/json; charset=utf-8",
            media_type="text/event-stream; charset=utf-8",
        )
    except Exception as e:
        return StreamingResponse(
            build_response_stream(
                client=request.client if request else "unknown", config="error", data=f"failed: {e}"
            ),
            status_code=400,
        )


def update_app_with_mcp(app: FastAPI) -> fastmcp.FastMCP:
    for ep in app.routes:
        if isinstance(ep, APIRoute):
            ep.operation_id = ep.name
    mcp = fastmcp.FastMCP.from_fastapi(app=app)
    # mcp_app = mcp.sse_app(path="/mcp", message_path="/mcp/messages")
    mcp_app = mcp.streamable_http_app(path="/mcp")
    app.state.mcp_app = mcp_app

    @asynccontextmanager
    async def app_lifespan(app: FastAPI):
        print(f"\n{' ':16} !!!! Bootup FastAPI server !!!! \n")
        app_config = AppConfig(model_opt=None)
        await app_config.setup_text_splitter()
        app.state.app_config = app_config
        app.state.chunkiter = app_config.chunkiter

        yield

        print(f"\n{' ':16} !!!! Shutdown FastAPI server !!!! \n")

    @asynccontextmanager
    async def lifespan_context(app: FastAPI):
        async with app_lifespan(app):
            async with mcp_app.lifespan(app):
                print(f"\n{' ':16} !!!! Starting MCP server !!!! \n")
                yield
                print(f"\n{' ':16} !!!! Closing MCP server !!!! \n")

    app.router.lifespan_context = lifespan_context
    app.mount("/v1/text/app", mcp_app)
    return mcp


if __name__ == "__main__":
    import uvicorn

    runner = update_app_with_mcp(app)
    print(app.routes)
    uvicorn.run(app, host="0.0.0.0", port=18002)
