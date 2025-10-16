import logging, os, json5, time, uuid, traceback, asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from typing import Dict, List
from load_config import AppConfig

from multi_closures_v3 import MultiClosureGroup, OutputMessage


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        print("\n\t!!!! startup !!!!\n")
        app_config = AppConfig(model_opt="local_llm:0")
        # app_config = AppConfig(model_opt="remote_llm:1")
        print("\n", app_config.model_specs, "\n")

        _app.state.app_config = app_config
        _app.state.closure_group = MultiClosureGroup(app_config=app_config)
        await app_config.setup_text_splitter()
    except Exception as e:
        raise RuntimeError(f"unable to startup. ({e})")

    yield

    try:
        print("\n\t!!!! shutdown !!!!\n")
        await _app.state.closure_group.close_session()
    except Exception as e:
        raise RuntimeError(f"unable to shutdown. ({e})")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn

    current_dir = os.path.dirname(os.path.abspath(__file__))
    uvicorn.run(
        "app_v3:app",
        host="0.0.0.0",
        port=35094,
        reload=False,
        reload_dirs=[current_dir],
    )


@app.post("/v1/text/summarize")
async def text_summarizations(request: Request):
    body: Dict = None
    session_id: str = None
    long_text: str = None
    app_config: AppConfig = app.state.app_config
    closure_group: MultiClosureGroup = app.state.closure_group
    logger: logging.Logger = app_config.logger
    try:
        body = await request.json()
        logger.debug(f"Request: {body}")
        session_id = body.get("session_id", "")
        long_text = body.get("long_text", "")
        threshold = body.get("threshold", 0)
        prompt_list = body.get("prompt_list", [])

        if not prompt_list:
            with open("configs/prompts.json", "r") as file:
                prompt_list = json5.load(file)

        if not long_text:
            raise Exception("missing long_text")
        if not isinstance(long_text, str):
            raise Exception("long_text must be string")

    except Exception as e:
        response = OutputMessage(
            session_id=session_id, type="ERROR", result=f"failed to handle session {session_id} ({e})"
        )
        return JSONResponse(response.to_dict(), status_code=200)

    session = await closure_group.register_closure_agents(session_id=session_id, prompts=prompt_list)

    if session.session_id == session_id:
        response = OutputMessage(
            session_id=session_id,
            type="ERROR",
            result=f"failed to double-run session {session.session_id}",
        )
        return JSONResponse(response.to_dict(), status_code=200)

    task_schedule = asyncio.create_task(
        closure_group.process_long_text(session_id=session.session_id, long_text=long_text, threshold=threshold)
    )

    response = OutputMessage(
        session_id=session.session_id,
        type="FINISH",
        result=f"succeed in starting session {session.session_id}! please check results later ({len(prompt_list)} totally)!!",
    )
    return JSONResponse(response.to_dict(), status_code=200)


@app.post("/v1/text/summarize/result")
async def text_summarizations_results(request: Request):
    body: Dict = None
    session_id: str = None
    stage_stream: bool = False
    app_config: AppConfig = app.state.app_config
    closure_group: MultiClosureGroup = app.state.closure_group
    logger: logging.Logger = app_config.logger

    try:
        body = await request.json()
        logger.debug(f"Request: {body}")
        session_id = body.get("session_id", "")
        stage_stream = body.get("stream", False)
        if not session_id:
            raise Exception("missing session_id")
        in_closure_group = closure_group
        session = closure_group.get_session(session_id=session_id)
        if not session:
            in_closure_group = closure_group
    except Exception as e:
        response = OutputMessage(
            session_id=session_id,
            type="ERROR",
            result=f"failed to retrieve results of session {session_id} ({e})",
        )
        return JSONResponse(response.to_dict(), status_code=200)

    if not stage_stream:
        one_result = await in_closure_group.retrieve_one_result(session_id=session_id, max_waits=3)
        return JSONResponse(one_result.to_dict(), status_code=200)
    else:
        return StreamingResponse(
            in_closure_group.stream_all_results(session_id),
            media_type="application/json; charset=utf-8",
        )


@app.post("/v1/text/summarize/dismiss")
async def text_summarizations_dismiss(request: Request):
    body: Dict = None
    session_id: str = None
    app_config: AppConfig = app.state.app_config
    closure_group: MultiClosureGroup = app.state.closure_group
    logger: logging.Logger = app_config.logger

    try:
        body = await request.json()
        logger.debug(f"Request: {body}")
        session_id = body.get("session_id", "")
        if not session_id:
            raise Exception("missing session_id")

        await closure_group.close_session(session_id=session_id)
        session = closure_group.get_session(session_id=session_id)
        if session:
            raise Exception(f"cannot stop {session_id}")
    except Exception as e:
        response = OutputMessage(
            session_id=session_id, type="ERROR", result=f"failed to close session {session_id} ({e})"
        )
        return JSONResponse(response.to_dict(), status_code=200)

    await asyncio.sleep(1)
    response = OutputMessage(
        session_id=session_id, type="FINISH", result=f"succeed in closing session {session_id}"
    )
    return JSONResponse(response.to_dict(), status_code=200)


@app.get("/v1/text/summarize/client")
async def index():
    return FileResponse("web_client.html")
