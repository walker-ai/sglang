# -*- encoding: utf-8 -*-
import json
import logging
import os
import socket
import sys
import traceback
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import numpy as np
import requests
import toml
import torch
import triton_python_backend_utils as pb_utils

# 组件使用文档，详见 https://yuque.antfin-inc.com/aii/aistudio/nkyse5
# python lib 依赖写入 requirement.txt

SERVER_ADDRESS = "http://127.0.0.1:9122"
STREAM_CHUNK_SIZE = 8192
STREAM_DELIMITER = b"\n\n"

DEFAULT_GPU_UTILIZATION = 0.88

DEFAULT_MAX_OUTPUT_LENGTH = 1024
SUPPORTED_MAX_OUTPUT_LENGTH = 4096
DEFAULT_BEAM_WIDTH = 1
DEFAULT_TEMPERATURE = 0.4
DEFAULT_TOP_K = 20
DEFAULT_TOP_P = 0.95
DEFAULT_DO_SAMPLE = False
STREAM_TIMEOUT_SECONDS = 120
TOKENIZER_TIMEOUT_SECONDS = 5

ENTRY_POINT_KEY = "__entry_point__"

logger = logging.getLogger()

OPEN_AI_CHAT_COMPLETION = "openai.chat.completion"
PROMPT_TOKENIZATION = "prompt.tokenization"
OPEN_AI_CHAT_TOKENIZATION = "openai.chat.tokenization"


class TritonPythonModel:
    """
        Your Python model must use the same class name. Every Python model
        that is created must have "TritonPythonModel" as the class name.
    """

    def __init__(self):
        self.module_dir = Path(__file__).parent.resolve()
        self.model_path = os.path.join(self.module_dir, 'autoupdate_resource', 'model')
        self.pid = os.getpid()
        self.hostname = socket.gethostname()

        self.default_handler = LegacyEntrypointRequestHandler()
        self.openai_handler = OpenAIEntrypointRequestHandler()

    def initialize(self, args):
        """
            `initialize` is called only once when the model is being loaded.
            Implementing `initialize` function is optional. This function allows
            the model to initialize any state associated with this model.
            Parameters
            ----------
            args : dict
            Both keys and values are strings. The dictionary keys and values are:
            * model_config: A JSON string containing the model configuration
            * model_instance_kind: A string containing model instance kind
            * model_instance_device_id: A string containing model instance device ID
            * model_repository: Model repository path
            * model_version: Model version
            * model_name: Model name
        """
        # 加载自定义配置文件
        custom_config = {}
        custom_config_path = os.path.join(self.module_dir, 'config.toml')
        if os.path.exists(custom_config_path):
            with open(custom_config_path, 'r') as f:
                custom_config = toml.load(f)
            logger.info(
                f'>>> Loaded `config.toml` file in {self.module_dir}, config={custom_config}'
            )

        # 写入 chat 模版
        if os.getenv("chat_template", ''):
            logger.info(">>> Used custom chat template <<<")
            logger.info(f"{os.getenv('chat_template')}")
            with open(os.path.join(self.module_dir, 'chat_template.jinja'), 'w') as f:
                f.write(os.getenv("chat_template"))

        # sglang 启动参数
        if os.getenv("sglang_boot_params") is not None:
            logger.info(">>> Using sglang_boot_params <<<")
            sglang_boot_args = json.loads(os.getenv("sglang_boot_params"))
        else:
            logger.info(">>> Using advanced config <<<")
            sglang_boot_args = {
                'dtype': os.getenv('dtype', 'bfloat16'),
                'mem_fraction_static': float(os.getenv('mem_fraction_static', DEFAULT_GPU_UTILIZATION)),
            }

        sglang_boot_args['tp_size'] = torch.cuda.device_count()

        # 判断是否存在 chat_template.jinja 对话模版文件，有则需要加载
        custom_chat_template_path = os.path.join(self.module_dir, 'chat_template.jinja')
        if os.path.exists(custom_chat_template_path):
            logger.info(f'>>> Loaded `chat_template.jinja` file in {self.module_dir}')
            sglang_boot_args['chat_template'] = custom_chat_template_path

        # 加载启动参数
        if custom_config.get("sglang") and custom_config.get("sglang").get("boot"):
            sglang_boot_args = {
                **sglang_boot_args,
                **custom_config["sglang"]["boot"],
            }

        # 加载环境变量
        if custom_config.get("envs"):
            envs = custom_config.get("envs")
            for k, v in envs.items():
                os.environ[k] = str(v)

        logger.info(f">>> [initialize] Starting SGLang server with "
                    f"model_path={self.model_path}, "
                    f"args = {sglang_boot_args}, "
                    f"envs = {os.environ}")

        sys.path.append('/home/admin/agent')
        import agent

        if not agent.try_wait_sglang_server(
            self.model_path, port=9122, **sglang_boot_args):
            raise Exception('>>> [initialize] Start SGLang server failed!!!')

        try:
            self.smoke_test()
        except Exception:
            logger.error(f"Smoke test failed! {traceback.format_exc()}")
            raise Exception("Smoke test failed!")

    def smoke_test(self):
        smoke_test_request = dict(
            model='auto',
            trace_id='trace_id',
            messages=[
                dict(
                    role='system',
                    content='You are a helpful assistant'
                ),
                dict(
                    role='user',
                    content='Hello?'
                )
            ],
            stream=False,
            # Here we set to 1, just for probing the service
            max_tokens=1
        )
        logger.info(f">>> [smoke_test] smoke test request: {smoke_test_request}")
        for output in self.openai_handler.chat_completion(smoke_test_request):
            logger.info(f">>> [smoke_test] smoke test output: {output}")

    def execute(self, inference_requests):
        """
            `execute` MUST be implemented in every Python model. `execute`
            function receives a list of pb_utils.InferenceRequest as the only
            argument. This function is called when an inference request is made
            for this model. Depending on the batching configuration (e.g. Dynamic
            Batching) used, `requests` may contain multiple requests. Every
            Python model, must create one pb_utils.InferenceResponse for every
            pb_utils.InferenceRequest in `requests`. If there is an error, you can
            set the error argument when creating a pb_utils.InferenceResponse

            Parameters
            ----------
            inference_requests : list
            A list of pb_utils.InferenceRequest

            Returns
            -------
            list
            A list of pb_utils.InferenceResponse. The length of this list must
            be the same as `requests`
        """
        try:
            maya_requests, request_ids = self._parse_requests(inference_requests)
            features = maya_requests[0]
            payload = json.loads(features.get("data").decode())
            logger.info(
                f">>> request={json.dumps(payload, ensure_ascii=False)}, request_id={request_ids[0]}")
            trace_id = request_ids[0]
        except Exception:
            error_msg = (
                f"Failed to parse payload from features, "
                f"inference_requests={inference_requests}, error={traceback.format_exc()}")
            yield self._make_error_response("none", error_msg)
            return

        # 版本兼容逻辑，默认为兼容旧版本的 LEGACY 模式，后续逐步切走
        entrypoint = payload.pop(ENTRY_POINT_KEY, None)
        if entrypoint is None:
            logger.warning(
                f"Using LEGACY entrypoint mode! features={features}")
            generator = self.default_handler.generate(payload)
        elif entrypoint == OPEN_AI_CHAT_COMPLETION:
            # 对于新版 entrypoint，只兼容 chat completion API，因为 completion API 已被 OpenAI 废弃
            generator = self.openai_handler.chat_completion(payload)
        # TODO(yudian.zy) 当前sglang还不支持tokenizer透出
        # elif entrypoint == PROMPT_TOKENIZATION:
        #     # 直接使用prompt正文进行编码
        #     generator = self.default_handler.tokenize(payload)
        # elif entrypoint == OPEN_AI_CHAT_TOKENIZATION:
        #     # 使用openai chat messages的协议进行编码
        #     generator = self.openai_handler.tokenize(payload)
        else:
            yield self._make_error_response(
                "none", f"Unsupported entrypoint type: {entrypoint}")
            return

        try:
            for res in generator:
                yield self._make_ok_response(res)
        except Exception:
            yield self._make_error_response(
                trace_id, f"Exception occurred when generating text:"
                          f" {traceback.format_exc()}")

    def _make_ok_response(self, res):
        maya_responses = [(0, 'ok', res)]
        return self._parse_responses(maya_responses)

    def _make_error_response(self, trace_id, error_msg):
        error_details = (
            f"Inference error: {error_msg}, "
            f"traceId={trace_id}, host={self.hostname}, pid={self.pid}")
        logger.error(error_details)
        result_code = -1
        result = {"result": error_details}
        maya_responses = [(result_code, "error", result)]
        return self._parse_responses(maya_responses)

    def _parse_requests(self, triton_requests):
        maya_requests = []
        request_ids = []
        for idx, infer_request in enumerate(triton_requests):
            id = infer_request.request_id()
            input_tensors = {}
            for input_tensor in infer_request.inputs():
                name = input_tensor.name()
                input = input_tensor.as_numpy()
                input = input[0]
                input_tensors[name] = input
            request_ids.append(id)
            maya_requests.append(input_tensors)
        return maya_requests, request_ids

    def _parse_responses(self, maya_responses):
        inference_responses = []
        for idx, response in enumerate(maya_responses):
            # If there is an error do not look into output
            if len(response) == 3 and response[0] != 0:
                # Maya UserHandler返回的code是用户自定义code(custom_code).0: 成功,其他: 失败
                # custom_code与TritonError枚举无法一一映射
                # 因此将custom_code与message一起放到json大字段中,由服务层返回时去解析
                error = {"message": response[1], "code": response[0]}
                inference_responses.append(
                    pb_utils.InferenceResponse(error=pb_utils.TritonError(
                        json.dumps(error))))
                continue
            output_tensors = response[2]
            response_tensors = []
            for output_idx, output_name in enumerate(output_tensors):
                output_value = output_tensors[output_name]
                if isinstance(output_value, np.ndarray):
                    output_np_array = output_value
                elif isinstance(output_value, list):
                    output_np_array = np.asarray(output_value)
                else:
                    output_np_array = np.asarray([output_value])

                response_tensors.append(
                    pb_utils.Tensor(output_name, output_np_array))
            inference_responses.append(
                pb_utils.InferenceResponse(output_tensors=response_tensors))
        return inference_responses


class OpenAIEntrypointRequestHandler:

    def __init__(self):
        self.api_base = SERVER_ADDRESS

    def tokenize(self, payload):
        trace_id = payload.get('trace_id', 'none')
        messages = payload.get('messages', None)
        if not messages:
            raise Exception(
                f"[{trace_id}] [OpenAIEntrypointRequestHandler][tokenize] Empty messages!"
            )
        yield self._do_tokenization(trace_id, messages)

    def chat_completion(self, payload):
        trace_id = payload.get('trace_id', 'none')

        # 在 trace_id 基础上增加随机数，防止同一个 trace 多次并行调用导致的 sglang 状态不一致
        # TODO(yudian.zy) 当前sglang还不支持传入自定义的request id
        sglang_trace_id = trace_id + '_' + str(uuid.uuid1()).replace('-', '')[:8]
        payload['trace_id'] = sglang_trace_id

        if not payload.get('top_k'):
            payload['top_k'] = DEFAULT_TOP_K

        top_p = payload.get('top_p', DEFAULT_TOP_P)
        if top_p >= 1.0:
            top_p = DEFAULT_TOP_P
        payload['top_p'] = top_p

        stream = payload.get('stream', True)
        payload['stream'] = stream

        if not payload.get('model', None):
            payload['model'] = 'auto'

        response = self._do_chat_completion(payload)
        logger.info(f"[{trace_id}] >>> Received response: {response}")

        # error handling
        if response.status_code != 200:
            # 如果返回了错误信息，直接返回
            if is_openai_error(response.text):
                res = {
                    "result": response.text,
                    # 以下参数均为兼容旧版本，实际值无任何意义！
                    "finished": False,
                    "finish_reason": "unknown",
                    "num_prompt_tokens": -1,
                    "num_generated_tokens": -1,
                    "num_cached_tokens": 0
                }
                logger.error(f"[{trace_id}] [OpenAIEntrypointRequestHandler][chat_completion]"
                             f" SGLang server returns non-200 status code: {response.status_code}, {response.text}, "
                             f"return res={res}")
                yield res
                return
            # 否则抛异常
            else:
                raise Exception(
                    f"[{trace_id}] [OpenAIEntrypointRequestHandler][chat_completion]"
                    f" SGLang server returns non-200 status code: {response.status_code}, {response.text}"
                )

        if stream:
            for chunk in response.iter_lines(chunk_size=STREAM_CHUNK_SIZE,
                                             decode_unicode=False,
                                             delimiter=STREAM_DELIMITER):
                if not chunk:
                    continue

                logger.info(f"[{trace_id}] >>> Iter chunk: {chunk}")

                # In SSE, each chunk should start with `data: `, so we skip the first 5 characters
                chunk = chunk.decode('utf-8')[5:].strip()
                # SSE end event
                if chunk == '[DONE]':
                    return

                res = {
                    "result": chunk or "",
                    # 以下参数均为兼容旧版本，实际值无任何意义！
                    "finished": False,
                    "finish_reason": "unknown",
                    "num_prompt_tokens": -1,
                    "num_generated_tokens": -1,
                    "num_cached_tokens": 0
                }
                logger.info(f"[{trace_id}] >>> Return res = {res}")
                yield res
        else:
            res = {
                "result": response.content.decode('utf-8') or "",
                # 以下参数均为兼容旧版本，实际值无任何意义！
                "finished": False,
                "finish_reason": "unknown",
                "num_prompt_tokens": -1,
                "num_generated_tokens": -1,
                "num_cached_tokens": 0
            }
            logger.info(f"[{trace_id}] >>> Return res = {res}")
            yield res

    def _do_completion(self, sglang_request) -> requests.Response:
        return self._post_http_request(
            url=self.api_base + '/v1/completions',
            request_body=sglang_request,
        )

    def _do_chat_completion(self, sglang_request) -> requests.Response:
        return self._post_http_request(
            url=self.api_base + '/v1/chat/completions',
            request_body=sglang_request,
        )

    def _do_tokenization(
        self,
        trace_id: str,
        content: Union[str, List[Dict[str, Any]]]
    ) -> requests.Response:
        # build request body
        req_body = {
            'model': 'auto',
            'trace_id': trace_id,
            'prompt' if isinstance(content, str) else 'messages': content,
        }
        # post
        response = self._post_http_request(
            url=self.api_base + '/tokenize',
            request_body=req_body,
            stream=False,
            timeout=TOKENIZER_TIMEOUT_SECONDS
        )

        # error handling
        if response.status_code != 200:
            # 如果返回了错误信息，直接返回
            if is_openai_error(response.text):
                res = {
                    "result": response.text,
                    # 以下参数均为兼容旧版本，实际值无任何意义！
                    "finished": False,
                    "finish_reason": "unknown",
                    "num_prompt_tokens": -1,
                    "num_generated_tokens": -1,
                    "num_cached_tokens": 0
                }
                logger.error(f"[{trace_id}] [OpenAIEntrypointRequestHandler][tokenize]"
                             f" SGLang server returns non-200 status code: {response.status_code}, {response.text}, "
                             f"return res={res}")
                yield res
                return
            # 否则抛异常
            else:
                raise Exception(
                    f"[{trace_id}] [OpenAIEntrypointRequestHandler][tokenize]"
                    f" SGLang server returns non-200 status code: {response.status_code}, {response.text}"
                )

        res = {
            "result": response.content.decode('utf-8'),
            # 以下参数均为兼容旧版本，实际值无任何意义！
            "finished": False,
            "finish_reason": "unknown",
            "num_prompt_tokens": -1,
            "num_generated_tokens": -1,
            "num_cached_tokens": 0
        }
        logger.info(f"[{trace_id}] >>> Return res = {res}")
        return res

    def _post_http_request(
        self,
        url: str,
        request_body: dict,
        stream: bool = True,
        timeout: int = STREAM_TIMEOUT_SECONDS
    ) -> requests.Response:
        headers = {"User-Agent": "Test Client"}
        # 工具问题特殊处理
        if not request_body.get("tools", None):
            request_body.pop("tool_choice", None)
        # 过滤空值
        request_body = {k: v for k, v in request_body.items() if v is not None}

        logger.info(f">>> Sending HTTP request to SGLang server: "
                    f"url={url}, headers={headers}, json={request_body}")

        return requests.post(
            url,
            headers=headers,
            json=request_body,
            stream=stream,
            timeout=timeout
        )


# 旧版本 API 服务兼容
class LegacyEntrypointRequestHandler(OpenAIEntrypointRequestHandler):

    def tokenize(self, payload):
        user_query = payload.get("query", "")
        chat_history = json.loads(payload.get("history", "[]"))
        system_prompt = payload.get("system_prompt", None)
        contains_template = self._is_query_contains_chat_template(user_query)
        if contains_template:
            yield self._do_tokenization(payload.get('trace_id', 'none'), user_query)
        else:
            messages = self._build_chat_completion_messages(
                user_query, system_prompt=system_prompt, chat_history=chat_history)
            yield self._do_tokenization(payload.get('trace_id', 'none'), messages)

    def generate(self, payload: Dict):
        trace_id = payload.get('trace_id') or 'none'

        user_query = payload.get("query", "")
        chat_history = json.loads(payload.get("history", "[]"))
        repetition_penalty = payload.get("repetition_penalty", None)
        system_prompt = payload.get("system_prompt", None)
        stop_sequences = payload.get("stop_sequences", [])
        max_output_length = payload.get("max_output_length",
                                        DEFAULT_MAX_OUTPUT_LENGTH)
        # 期望最大输出超长后的限制
        if (max_output_length is not None and
                max_output_length > SUPPORTED_MAX_OUTPUT_LENGTH):
            logger.info(f"[{trace_id}] decrease max_output_length(max_tokens) "
                        f"from {max_output_length} to {SUPPORTED_MAX_OUTPUT_LENGTH}.")
            max_output_length = SUPPORTED_MAX_OUTPUT_LENGTH

        temperature = payload.get("temperature", DEFAULT_TEMPERATURE)
        top_k = payload.get("top_k", DEFAULT_TOP_K)
        top_p = payload.get("top_p", DEFAULT_TOP_P)

        if top_p >= 1.0:
            top_p = DEFAULT_TOP_P

        do_sample = payload.get("do_sample", DEFAULT_DO_SAMPLE)
        if not do_sample:
            temperature = 0.0
            top_k = -1
            top_p = 1.0

        is_sync = payload.get("sync", True)
        stream = not is_sync

        contains_template = self._is_query_contains_chat_template(user_query)
        # 如果输入的 query 已经包含 Chat 模版，则使用 OpenAI Completion 接口进行推理
        if contains_template:
            return self._generate_with_completion(
                trace_id,
                prompt=user_query,
                stream=stream,
                max_output_length=max_output_length,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                stop_sequences=stop_sequences,
                top_p=top_p,
                top_k=top_k,
            )
        # 否则，使用 ChatCompletion 接口进行推理
        else:
            return self._generate_with_chat_completion(
                trace_id,
                user_query,
                system_prompt=system_prompt,
                chat_history=chat_history,
                stream=stream,
                max_output_length=max_output_length,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                stop_sequences=stop_sequences,
                top_p=top_p,
                top_k=top_k)

    def _generate_with_chat_completion(
        self,
        trace_id,
        user_query,
        system_prompt=None,
        chat_history=None,
        stream=False,
        max_output_length=DEFAULT_MAX_OUTPUT_LENGTH,
        temperature=DEFAULT_TEMPERATURE,
        repetition_penalty=None,
        stop_sequences=None,
        top_p=DEFAULT_TOP_P,
        top_k=DEFAULT_TOP_K):
        logger.info(
            f"[{trace_id}] >>> LegacyEntrypointRequestHandler._generate_with_chat_completion"
        )
        messages = self._build_chat_completion_messages(
            user_query, system_prompt=system_prompt, chat_history=chat_history)

        sglang_trace_id = trace_id + "_" + str(uuid.uuid1()).replace('-', '')[:8]

        # 底层使用 OpenAI 兼容的 ChatCompletion API
        sglang_request = {
            "messages": messages,
            "model": "auto",
            "max_tokens": max_output_length,
            "n": 1,
            "stream": stream,
            "temperature": temperature,
            "stop": stop_sequences,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "top_k": top_k,
            "trace_id": sglang_trace_id
        }

        if stream:
            sglang_request['stream_options'] = dict(
                include_usage=True
            )

        response = self._do_chat_completion(sglang_request)

        logger.info(f"[{trace_id}]>>> Received response = {response}")

        # error handling
        if response.status_code != 200:
            raise Exception(
                f"[{trace_id}][LegacyEntrypointRequestHandler][_generate_with_chat_completion] "
                f"SGLang server returns non-200 status code: {response.status_code}, {response.text}"
            )

        if stream:
            finish_reason = None
            for chunk in response.iter_lines(chunk_size=STREAM_CHUNK_SIZE,
                                             decode_unicode=False,
                                             delimiter=STREAM_DELIMITER):
                if not chunk:
                    continue

                logger.info(f"[{trace_id}] >>> Iter chunk data = {chunk}")

                # In SSE, each chunk should start with `data: `, so we skip the first 5 characters
                chunk = chunk.decode('utf-8')[5:].strip()
                # SSE end event
                if chunk == '[DONE]':
                    logger.info(f"[{trace_id}] >>> DONE")
                    return

                ret_data = json.loads(chunk)
                if ret_data.get("object") == 'error':
                    raise Exception(
                        f"[{trace_id}] ERROR: errorType={ret_data.get('type')}, "
                        f"errorCode={ret_data.get('code')}, errorMsg={ret_data.get('message')}"
                    )

                choices = ret_data.get('choices')
                content = ''
                if choices:
                    choice = choices[0]
                    finish_reason = choice.get('finish_reason')
                    content = choice.get('delta').get('content')

                res = {
                    "result": content or "",
                    "finished": finish_reason is not None,
                    "finish_reason": finish_reason or "unknown",
                    "num_prompt_tokens": -1,
                    "num_generated_tokens": -1,
                    "num_cached_tokens": 0
                }

                usage = ret_data.get('usage')
                if usage:
                    res['num_prompt_tokens'] = usage.get("prompt_tokens", -1)
                    res['num_generated_tokens'] = usage.get("completion_tokens", -1)
                    # 前缀命中情况
                    prompt_tokens_details = usage.get("prompt_tokens_details", None)
                    if prompt_tokens_details:
                        res['num_cached_tokens'] = prompt_tokens_details.get("cached_tokens", 0)

                logger.info(f"[{trace_id}] >>> Return res = {res}")
                yield res
        else:
            ret_data = json.loads(response.content)
            logger.info(f"[{trace_id}] >>> Received ret_data = {ret_data}")

            if ret_data.get("object") == 'error':
                raise Exception(
                    f"[{trace_id}] ERROR: errorType={ret_data.get('type')}, "
                    f"errorCode={ret_data.get('code')}, errorMsg={ret_data.get('message')}"
                )

            choices = ret_data.get('choices')
            choice = choices[0]
            content = choice.get('message').get('content')
            finish_reason = choice.get('finish_reason')

            res = {
                "result": content or "",
                "finished": finish_reason is not None,
                "finish_reason": finish_reason or "unknown",
                "num_prompt_tokens": -1,
                "num_generated_tokens": -1,
                "num_cached_tokens": 0
            }

            usage = ret_data.get('usage')
            if usage:
                res['num_prompt_tokens'] = usage.get("prompt_tokens", -1)
                res['num_generated_tokens'] = usage.get("completion_tokens", -1)
                # 前缀命中情况
                prompt_tokens_details = usage.get("prompt_tokens_details", None)
                if prompt_tokens_details:
                    res['num_cached_tokens'] = prompt_tokens_details.get("cached_tokens", 0)

            logger.info(f"[{trace_id}] >>> Return res = {res}")

            yield res

    def _generate_with_completion(self,
                                  trace_id,
                                  prompt,
                                  stream=False,
                                  max_output_length=DEFAULT_MAX_OUTPUT_LENGTH,
                                  temperature=DEFAULT_TEMPERATURE,
                                  repetition_penalty=None,
                                  stop_sequences=None,
                                  top_p=DEFAULT_TOP_P,
                                  top_k=DEFAULT_TOP_K):

        sglang_trace_id = trace_id + "_" + str(uuid.uuid1()).replace('-', '')[:8]

        # 底层使用 OpenAI 兼容的 completion API
        sglang_request = {
            "model": "auto",
            "prompt": prompt,
            "stream": stream,
            "n": 1,
            "max_tokens": max_output_length,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "stop": stop_sequences,
            "top_p": top_p,
            "top_k": top_k,

            # extra params
            "trace_id": sglang_trace_id,
        }

        if stream:
            sglang_request['stream_options'] = dict(
                include_usage=True
            )

        response = self._do_completion(sglang_request)

        logger.info(f"[{trace_id}] >>> Received response={response}")

        # error handling
        if response.status_code != 200:
            raise Exception(
                f"[{trace_id}][LegacyEntrypointRequestHandler][_generate_with_completion] "
                f"SGLang server returns non-200 status code: {response.status_code}, {response.text}"
            )

        if stream:
            finish_reason = None
            for chunk in response.iter_lines(
                chunk_size=STREAM_CHUNK_SIZE,
                decode_unicode=False,
                delimiter=STREAM_DELIMITER,
            ):
                if not chunk:
                    continue

                logger.info(f"[{trace_id}] >>> Iter chunk")

                # In SSE, each chunk should start with `data: `, so we skip the first 5 characters
                chunk = chunk.decode('utf-8')[5:].strip()
                # SSE end event
                if chunk == '[DONE]':
                    return

                ret_data = json.loads(chunk)

                choices = ret_data.get('choices')
                content = ''
                if choices:
                    choice = choices[0]
                    content = choice.get('text')
                    finish_reason = choice.get('finish_reason')

                # 兼容现有格式
                res = {
                    "result": content or "",
                    "finish_reason": finish_reason or "unknown",
                    "finished": finish_reason is not None,
                    "num_prompt_tokens": -1,
                    "num_generated_tokens": -1,
                    "num_cached_tokens": 0
                }

                usage = ret_data.get('usage')
                if usage:
                    res['num_prompt_tokens'] = usage.get('prompt_tokens', -1)
                    res['num_generated_tokens'] = usage.get('completion_tokens', -1)
                    # 前缀命中情况
                    prompt_tokens_details = usage.get("prompt_tokens_details", None)
                    if prompt_tokens_details:
                        res['num_cached_tokens'] = prompt_tokens_details.get("cached_tokens", 0)

                logger.info(f"[{trace_id}] >>> Return res={res}")
                yield res
        else:
            ret_data = json.loads(response.content)
            choices = ret_data.get('choices')
            choice = choices[0]
            content = choice['text']
            finish_reason = choice.get('finish_reason')

            usage = ret_data.get('usage') or {}

            # 前缀命中情况
            num_cached_tokens = 0
            prompt_tokens_details = usage.get("prompt_tokens_details", None)
            if prompt_tokens_details:
                num_cached_tokens = prompt_tokens_details.get("cached_tokens", 0)

            # 兼容现有格式
            res = {
                "result": content or "",
                "finish_reason": finish_reason or "unknown",
                "finished": True,
                "num_prompt_tokens": usage.get('prompt_tokens', -1),
                "num_generated_tokens": usage.get('completion_tokens', -1),
                "num_cached_tokens": num_cached_tokens
            }

            logger.info(f"[{trace_id}] >>> Return res={res}")
            yield res

    def _is_query_contains_chat_template(self, query: str) -> bool:
        # 百灵 ChatML
        if (('<role>HUMAN</role>' in query)
            or ('<role>SYSTEM</role>' in query)):
            return True

        # 通义千问
        if (('<|im_start|>user' in query)
            or ('<|im_start|>system' in query)):
            return True

        # llama2
        if "[INST]" in query or '[/INST]' in query:
            return True

        # llama3
        if (('<|start_header_id|>user<|end_header_id|>' in query)
            or ('<|start_header_id|>system<|end_header_id|>' in query)):
            return True

        # ChatGLM
        if ('<|system|>' in query) or ('<|user|>' in query):
            return True

        return False

    def _build_chat_completion_messages(self, user_query: str,
                                        system_prompt: Optional[str],
                                        chat_history: Optional[List]):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if chat_history:
            for dialog in chat_history:
                user = dialog["user"].strip()
                bot = dialog["bot"].strip()

                messages.append({"role": "user", "content": user})
                messages.append({"role": "assistant", "content": bot})

        if user_query:
            messages.append({"role": "user", "content": user_query})
        return messages


def is_openai_error(response: str) -> bool:
    if not response:
        return False
    try:
        response_json = json.loads(response)
        if response_json['object'] == 'error':
            return True
    except Exception as e:
        return False
