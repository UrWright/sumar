import logging, asyncio, uuid, time, re, json, inspect
from markdown_it import MarkdownIt
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import (
    Callable,
    Literal,
    Optional,
    AsyncGenerator,
    List,
    Dict,
    Tuple,
    Awaitable,
    Any,
    Sequence,
    Union,
)
from load_config import AppConfig

from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core import (
    ClosureAgent,
    ClosureContext,
    DefaultTopicId,
    MessageContext,
    AgentId,
    AgentType,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
    DefaultInterventionHandler,
    DropMessage,
)
from autogen_core.tools import ToolResult, ToolSchema, Tool
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    CreateResult,
    UserMessage,
    SystemMessage,
)
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools import mcp
from mcp.types import TextContent

"""
from multi_closures import (
    TextInputMessage,
    OutputMessage,
    PromptItem,
    SessionItem,
    BaseClosureGroup,
    BaseClosureGroupV2,
    BaseClosureGroupV3,
)
"""


@dataclass_json
@dataclass
class TextInputMessage:
    session_id: str
    threshold: int = 0
    long_text: str = ""


@dataclass_json
@dataclass
class OutputMessage:
    session_id: str
    type: str
    result: str


@dataclass
class PromptItem:
    purpose: str
    system_prompt: str
    queue: Optional[asyncio.Queue] = None
    agent: Optional[AgentId] = None
    model_client: Optional[OpenAIChatCompletionClient] = None


@dataclass
class SessionItem:
    session_id: str
    runtime: SingleThreadedAgentRuntime
    queue: asyncio.Queue
    prompts: List[PromptItem]
    counter: int = 0
    agents: List[AgentId] = None
    context: BufferedChatCompletionContext = None
    model_client: Optional[OpenAIChatCompletionClient] = None

    def increment_counter(self):
        self.counter += 1
        return self.counter


CLOSE_QUEUE = object()


class MultiClosureGroup:
    def __init__(self, app_config: AppConfig, prefix: str = "meeting"):
        self.app_config = app_config
        self._logger = self.app_config.logger

        self._prefix = prefix
        self._topic_types: Dict[str, str] = {
            "chunkit": f"{prefix}_chunkit",
            "process": f"{prefix}_process",
            "result": f"{prefix}_result",
        }
        self._agent_prefix = "at"
        self._session_list: List[SessionItem] = []
        self._session_index: dict[str, SessionItem] = {}

    def _topic_id_of(self, title: str, session_id: str = "default") -> TopicId:
        topic_type_name = self._topic_types.get(title, None)
        if not topic_type_name:
            return None
        topic_id = TopicId(type=topic_type_name, source=session_id)
        return topic_id

    def _agent_type_of(self, topic_type_name: str, purpose: str = ""):
        agent_type_name = f"{self._agent_prefix}_{topic_type_name}" + f'_{purpose if purpose != "" else ""}'
        return agent_type_name

    def _agent_id_of(self, topic_type_name: str, purpose: str = "", session_id: str = "default") -> AgentId:
        agent_type_name = self._agent_type_of(topic_type_name=topic_type_name, purpose=purpose)
        agent_id = AgentId(type=agent_type_name, key=session_id)
        return agent_id

    def _new_session(self, session_id, prompts: List) -> SessionItem:
        session_prompts: List[PromptItem] = [PromptItem(**p, queue=asyncio.Queue()) for p in prompts]
        session = SessionItem(
            session_id=session_id,
            runtime=SingleThreadedAgentRuntime(
                intervention_handlers=[DefaultInterventionHandler()],
                ignore_unhandled_exceptions=False,
            ),
            agents=[],
            queue=asyncio.Queue(),
            prompts=session_prompts,
            context=BufferedChatCompletionContext(len(session_prompts)),
        )
        return session

    def add_session(self, session_id: str, prompts: List[Dict]) -> SessionItem:
        session = self._new_session(session_id, prompts)
        self._session_list.append(session)
        self._session_index[session_id] = session
        return session

    def get_session(self, session_id: str) -> SessionItem | None:
        return self._session_index.get(session_id)

    async def _drop_session_messages(self, session: SessionItem):
        self._logger.debug(f"session[{session.session_id}] drop messages")
        for prompt in session.prompts:
            if prompt.agent is not None:
                await session.runtime.send_message(message=DropMessage(), recipient=prompt.agent)
        for agent in session.agents:
            await session.runtime.send_message(message=DropMessage(), recipient=agent)

        for key, val in self._topic_types.items():
            topic_id = self._topic_id_of(title=val, session_id=session.session_id)
            if topic_id is not None:
                await session.runtime.publish_message(message=DropMessage(), topic_id=topic_id)

    async def _release_session_prompts(self, session: SessionItem):
        self._logger.debug(f"session[{session.session_id}] release prompts")
        for prompt in session.prompts:
            if prompt.model_client:
                await prompt.model_client.close()
                prompt.model_client = None
                if prompt.queue:
                    await prompt.queue.put(CLOSE_QUEUE)

    async def _release_session(self, session: SessionItem):
        if session and session.runtime:
            self._logger.debug(f"session[{session.session_id}] close runtime@{hex(id(session.runtime))} ")

            await self._release_session_prompts(session)
            await self._drop_session_messages(session)

            if session.context:
                await session.context.clear()
            await session.runtime.close()
            await session.queue.put(CLOSE_QUEUE)

            session.runtime = None
            self._session_list.remove(session)
            self._session_index.pop(session.session_id, None)

    async def close_session(self, session_id: str | None = None):
        self._logger.info(f"\n\n!!!! closing session[{session_id}] !!!!\n\n")
        if session_id is not None:
            session = self.get_session(session_id)
            await self._release_session(session)
        else:
            for s_id, session in list(self._session_index.items()):
                await self._release_session(session)
            self._session_list.clear()
            self._session_index.clear()

    def build_output_message(self, session_id, type, result: CreateResult | str) -> OutputMessage:
        if isinstance(result, CreateResult):
            content = result.content
        elif isinstance(result, str):
            content = result
        else:
            raise TypeError(f"ERROR: unsupported result data type [{type(result)}]")

        return OutputMessage(session_id=session_id, type=type, result=content)

    async def _remote_split_text_to_segments(
        self, message: TextInputMessage, ctx: MessageContext
    ) -> List[TextInputMessage]:
        server_params = mcp.StreamableHttpServerParams(
            url="http://127.0.0.1:18002/v1/text/app/mcp",
            headers={
                "content-type": "application/json",
                "Accept": "text/event-stream, application/json",
            },
            timeout=30,
        )

        mcp_workbench = mcp.McpWorkbench(server_params=server_params)
        tool_name = "split_text_to_segments"
        tool_args = {
            "client": message.session_id,
            "config": f'"threshold={message.threshold}"',
            "data": message.long_text,
        }
        tool_result = None
        message_list = []
        try:
            await mcp_workbench.start()
            tool_result = await mcp_workbench.call_tool(
                name=tool_name, arguments=tool_args, cancellation_token=ctx.cancellation_token
            )
            for result in tool_result.result:
                content_list = result.content.splitlines()
                for content in content_list:
                    if isinstance(content, str):
                        content: Dict = json.loads(content)
                        data: Dict = content.get("data", "")
                        message_list.append(
                            TextInputMessage(session_id=message.session_id, long_text=data.get("text"))
                        )
        except asyncio.CancelledError:
            self._logger.error(f"call MCP SPLITER cancelled by runtime !!")
            message_list = []
        except Exception as e:
            self._logger.error(f"call MCP SPLITER failed: {e}")
            message_list = []
        finally:
            await mcp_workbench.stop()
        return message_list

    async def _local_split_text_to_segments(self, message: TextInputMessage) -> List[TextInputMessage]:
        message_list = []
        try:
            chunkiter = self.app_config.chunkiter
            text_tokens = await chunkiter.count_tokens(message.long_text)
            if text_tokens <= message.threshold:
                return [message]

            sentences = await chunkiter.text_to_sentences(message.long_text)
            async for chunk in chunkiter.sentences_to_chunks(sentences, message.threshold):
                message_list.append(TextInputMessage(session_id=message.session_id, long_text=chunk.text))

        except asyncio.CancelledError:
            self._logger.error(f"local SPLITER cancelled by runtime !!")
            message_list = []
        except Exception as e:
            self._logger.error(f"local SPLITER failed: {e}")
            message_list = []

        self._logger.info(f"!! Return the list of messages (len={len(message_list)})!!")
        return message_list

    # T_SplitterModelProvider = Union[
    #    Callable[[SessionItem], Awaitable[List[TextInputMessage]]],
    #    Callable[[SessionItem, MessageContext], Awaitable[List[TextInputMessage]]],
    # ]

    def create_checkit_message_id(self, session: SessionItem):
        message_id = f"chunkit@{session.session_id}/{hex(id(session.runtime))}"
        return message_id

    async def _create_closure_chunkit_handler(
        self, session: SessionItem, model_provider: Literal["local", "remote"]
    ) -> Callable[
        [ClosureContext, TextInputMessage, MessageContext],
        Awaitable[Any],
    ]:
        topic_type_name = self._topic_types["process"]
        topic_id = TopicId(type=topic_type_name, source=session.session_id)
        runtime = session.runtime
        session_id = session.session_id

        async def chunkit_closure_handler(
            agent: ClosureContext, message: TextInputMessage, ctx: MessageContext
        ) -> Any:  # List[TextInputMessage]:
            self._logger.debug(f"agent[{agent.id}] chunkit:: threshold=[{message.threshold}]")
            if session_id != message.session_id:
                raise Exception(
                    f"unmatched session_id (session vs message: {session_id} vs {message.session_id})"
                )
            if message.threshold > 512:
                if model_provider == "local":
                    message_list = await self._local_split_text_to_segments(message)
                elif model_provider == "remote":
                    message_list = await self._remote_split_text_to_segments(message, ctx)
                else:
                    raise Exception(f"unsupported model provider [{model_provider}]")
                if not message_list:
                    raise Exception(f"invalid/empty message list after chunk")

            else:
                message_list = [TextInputMessage(session_id=session_id, long_text=message.long_text)]

            msg_id = self.create_checkit_message_id(session)
            msg = TextInputMessage(session_id=session_id, long_text=f"chunks={len(message_list)}")
            self._logger.debug(f"agent[{agent.id}] publish header: id={msg_id}, content={msg}")
            await runtime.publish_message(
                message=msg,
                topic_id=topic_id,
                cancellation_token=ctx.cancellation_token,
                sender=agent.id,
                message_id=msg_id,
            )
            for i, msg in enumerate(message_list):
                self._logger.debug(f"agent[{agent.id}] publish chunk[{i}]\n{msg}")
                await runtime.publish_message(
                    message=msg,
                    topic_id=topic_id,
                    cancellation_token=ctx.cancellation_token,
                    sender=agent.id,
                    message_id=None,
                )

        return chunkit_closure_handler

    async def _create_closure_process_handler(
        self, session: SessionItem, index: int, model_provider: Dict
    ) -> Callable[[ClosureContext, TextInputMessage, MessageContext], Awaitable[Optional[OutputMessage]]]:

        prompt: PromptItem = session.prompts[index]
        purpose = prompt.purpose
        system_prompt = prompt.system_prompt
        session_id = session.session_id
        context = session.context
        runtime = session.runtime
        total_chunks = 0
        remain_chunks = 0
        done_chunks = 0
        total_result = ""

        async def process_closure_handler(
            agent: ClosureContext, message: TextInputMessage, ctx: MessageContext
        ) -> Optional[OutputMessage]:

            # self._logger.debug(f"agent[{agent.id}] process input[{purpose}]")

            output_message = None

            nonlocal total_chunks
            nonlocal remain_chunks
            nonlocal done_chunks
            nonlocal total_result

            begin_time = time.perf_counter()

            # try:
            if session_id != message.session_id:
                raise RuntimeError(
                    f"unmatched session_id (session vs message: {session_id} vs {message.session_id})"
                )

            # model_client = prompt.model_client
            model_client = session.model_client

            if not model_client:
                self._logger.debug(f"agent[{agent.id}] connect to LLM ")
                model_client = OpenAIChatCompletionClient(**model_provider)

            # prompt.model_client = model_client
            session.model_client = model_client

            if ctx.message_id == self.create_checkit_message_id(session):
                _total_chunks = int(message.long_text.split("=")[1])
                self._logger.debug(f"agent[{agent.id}] will process totally chunks[{_total_chunks}]")

                if total_chunks != 0:
                    raise Exception(
                        f"repeated notificaitoin of total chunks ({_total_chunks}), previous={total_chunks}"
                    )
                total_chunks = _total_chunks
                total_result = (
                    f"There are {total_chunks} parts of information about {purpose} of the {self._prefix}. \n\n"
                )
                return None

            remain_chunks += 1
            self._logger.debug(f"agent[{agent.id}] receive chunk[{remain_chunks}/{total_chunks}]")

            if remain_chunks > total_chunks or done_chunks > total_chunks:
                raise Exception(
                    f"invalid chunk number: total={total_chunks}, done={done_chunks}, remain={remain_chunks}"
                )

            sys_message = SystemMessage(content=system_prompt)
            user_message = UserMessage(content=message.long_text, source=session_id)
            message_list = [sys_message, user_message]

            model_response: Optional[CreateResult] = await model_client.create(
                messages=message_list, cancellation_token=ctx.cancellation_token
            )

            if model_response is None:
                raise RuntimeError("failed to call llm: no model result")

            done_chunks += 1
            if total_chunks > 1:
                total_result += f"# Part {done_chunks}: \n\n{model_response.content} \n\n"
            else:
                total_result = model_response.content

            self._logger.debug(f"agent[{agent.id}] done chunk[{done_chunks}/{total_chunks}]")
            if done_chunks == total_chunks:
                output_message = self.build_output_message(
                    session_id=session_id,
                    type=purpose,
                    result=total_result,
                )
                # prompt.queue.put_nowait(output_message)

                # topic_id = self._topic_id_of(title="result", session_id=session_id)
                recipient = self._agent_id_of(topic_type_name=self._topic_types["result"])
                self._logger.debug(
                    f"agent[{agent.id}] notify result agent[{recipient}]: done=({done_chunks}/{total_chunks})\n{output_message}"
                )
                await runtime.send_message(
                    message=output_message,
                    recipient=recipient,
                    sender=agent.id,
                    cancellation_token=ctx.cancellation_token,
                    message_id=f"chunks={total_chunks}",
                )
                # await prompt_item.model_client.close()

                end_time = time.perf_counter()
                # print(f"\n\n {purpose} end time: {end_time} s (elapse {end_time - begin_time}) s\n\n")

            return output_message

        return process_closure_handler

    async def _create_closure_result_handler(
        self, session: SessionItem
    ) -> Callable[[ClosureContext, OutputMessage, MessageContext], Awaitable[None]]:
        model_provider = self.app_config.model_provider
        queue = session.queue
        session_id = session.session_id
        sys_prompt = ""
        sys_message = None

        async def result_closure_handler(
            agent: ClosureContext, message: OutputMessage, ctx: MessageContext
        ) -> None:
            nonlocal sys_prompt
            nonlocal sys_message
            if session_id != message.session_id:
                raise RuntimeError(
                    f"unmatched session_id (session vs message: {session_id} vs {message.session_id})"
                )

            if not session.model_client:
                self._logger.debug(f"agent[{agent.id}] connect to LLM ")
                session.model_client = OpenAIChatCompletionClient(**model_provider)

            sys_prompt = f"""
You are an AI assistant.
You will receive one or more pieces of information about the {message.type} of {self._prefix}.
Your task is to combine all parts into a single, coherent result, removing any redundancy or overlap.
Ensure the final output is clear, concise, and logically consistent.
""".strip()
            sys_message = SystemMessage(content=sys_prompt)
            user_message = UserMessage(content=message.result, source=session.session_id)

            model_response = await session.model_client.create(
                [sys_message, user_message], cancellation_token=ctx.cancellation_token
            )
            if model_response is None:
                raise RuntimeError("failed to call llm: no model result")

            output_message = self.build_output_message(
                session_id=session.session_id,
                type=message.type,
                result=model_response,
                # result=model_response.content,
            )

            self._logger.debug(f"agent[{agent.id}] put output[{message.type}] to final queue")
            queue.put_nowait(output_message)
            return

        return result_closure_handler

    async def _register_closure_chunkit_agent(
        self, session: SessionItem, model_provider: Literal["local", "remote"]
    ) -> AgentType:

        topic_type_name = self._topic_types["chunkit"]

        agent_closure_chunkit_handler = await self._create_closure_chunkit_handler(session, model_provider)

        agent_type_name = self._agent_type_of(topic_type_name=topic_type_name)
        self._logger.debug(f"agent type[{agent_type_name}/{topic_type_name}]")
        agent_type = await ClosureAgent.register_closure(
            runtime=session.runtime,
            type=agent_type_name,
            closure=agent_closure_chunkit_handler,
            subscriptions=lambda: [TypeSubscription(topic_type=topic_type_name, agent_type=agent_type_name)],
        )
        return agent_type

    async def _register_closure_process_agents(self, session: SessionItem):
        # topic_type_name = self._topic_id_of(title="process", session_id=session.session_id).type
        topic_type_name = self._topic_types["process"]
        agent_num = 0
        for idx, prompt in enumerate(session.prompts):

            agent_closure_process_handler = await self._create_closure_process_handler(
                session=session, index=idx, model_provider=self.app_config.model_provider
            )
            agent_type_name = self._agent_type_of(topic_type_name=topic_type_name, purpose=prompt.purpose)
            agent_type: AgentType = await ClosureAgent.register_closure(
                runtime=session.runtime,
                type=agent_type_name,
                closure=agent_closure_process_handler,
                subscriptions=lambda: [
                    TypeSubscription(topic_type=topic_type_name, agent_type=agent_type_name)
                ],
                unknown_type_policy="ignore",
            )
            # prompt.agent = AgentId(agent_type, key="default")
            agent_num += 1

    async def _register_closure_result_agent(self, session: SessionItem):
        topic_type_name = self._topic_types["result"]

        agent_closure_result_handler = await self._create_closure_result_handler(session)
        agent_type_name = self._agent_type_of(topic_type_name=topic_type_name)
        agent_type = await ClosureAgent.register_closure(
            runtime=session.runtime,
            type=agent_type_name,
            closure=agent_closure_result_handler,
            subscriptions=lambda: [TypeSubscription(topic_type=topic_type_name, agent_type=agent_type_name)],
        )
        # session.agent = AgentId(agent_type, key="default")

    async def register_closure_agents(self, session_id: str, prompts: Optional[List[Dict]]) -> SessionItem:
        async def release_closure_runtime(session: SessionItem):
            while True:
                if not session.runtime:
                    self._logger.debug(f"session[{session_id}] already closed")
                    break
                if session.counter == len(session.prompts):
                    self._logger.debug(f"session[{session_id}] queue pulled to empty")
                    await self.close_session(session_id=session.session_id)
                await asyncio.sleep(1)

        session = self.get_session(session_id)
        if session:
            self._logger.info(f"session[{session_id}] exists (no registration needed)")
            return session

        new_session_id = str(uuid.uuid4()).replace("-", "")
        session = self.add_session(session_id=new_session_id, prompts=prompts)

        agent_type = await self._register_closure_chunkit_agent(session=session, model_provider="local")
        await self._register_closure_process_agents(session)
        await self._register_closure_result_agent(session)

        session.runtime.start()
        task_close = asyncio.create_task(release_closure_runtime(session=session))
        return session

    async def process_long_text(self, session_id: str, long_text: str, threshold: int):
        session = self.get_session(session_id=session_id)
        session.counter = 0
        input_message = TextInputMessage(session_id=session_id, long_text=long_text, threshold=threshold)
        await session.runtime.publish_message(
            message=input_message, topic_id=self._topic_id_of(title="chunkit", session_id=session_id)
        )
        return

    async def retrieve_one_result(self, session_id: str, max_waits):
        session = self.get_session(session_id=session_id)
        self._logger.debug(session)
        if not session:
            return OutputMessage(
                session_id=session_id,
                type="ERROR",
                result=f"️failed to retrieve a result of session {session_id} (session not started yet)",
            )

        total_prompts = len(session.prompts)

        if session.counter == total_prompts:
            return OutputMessage(
                session_id=session_id,
                type="FINISH",
                result=f"️finish retrieving all {total_prompts} results (Done? {session.queue.empty()})",
            )

        wait_counter = 0
        while wait_counter < max_waits:
            try:
                output_message: OutputMessage = session.queue.get_nowait()
                if output_message == CLOSE_QUEUE:
                    response = OutputMessage(
                        session_id=session_id,
                        type="ERROR",
                        result=f"️failed as session {session_id} dismissed",
                    )
                    return response
                self._logger.debug(f"+ Type: {output_message.type} ")
                self._logger.debug(f"+ Session: {output_message.session_id}")
                self._logger.debug(f"+ Content: {output_message.result}")
                session.increment_counter()
                return output_message
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.5)
                wait_counter += 1

        return OutputMessage(
            session_id=session_id,
            type="ERROR",
            result=f"️unable to retrieve results[{session.counter}/{total_prompts}] (tried for {max_waits} times) ",
        )

    async def stream_all_results(self, session_id: str):
        session = self.get_session(session_id=session_id)
        if not session:
            response = OutputMessage(
                session_id=session_id,
                type="ERROR",
                result=f"️failed to retrieve all results of session {session_id} (session not started yet)",
            )
            yield response.to_json(ensure_ascii=False) + "\n"
            return

        total_prompts = len(session.prompts)

        while session.counter < total_prompts:
            try:
                output_message = session.queue.get_nowait()
                if output_message == CLOSE_QUEUE:
                    response = OutputMessage(
                        session_id=session_id,
                        type="ERROR",
                        result=f"️failed as session {session_id} dismissed",
                    )
                    yield response.to_json(ensure_ascii=False) + "\n"
                    return
                session.increment_counter()
                self._logger.debug(f"+ Type: {output_message.type} ")
                self._logger.debug(f"+ Session: {output_message.session_id}")
                self._logger.debug(f"+ Content: {output_message.result}")

                yield output_message.to_json(ensure_ascii=False) + "\n"
            except asyncio.QueueEmpty:
                # self._logger.debug(f"////// waiting for results[{session.counter}/{total_prompts}] \n")
                await asyncio.sleep(0.5)

        response = OutputMessage(
            session_id=session_id,
            type="FINISH",
            result=f"️finish retrieving all {total_prompts} results (Done? {session.queue.empty()})",
        )
        yield response.to_json(ensure_ascii=False) + "\n"
        return
