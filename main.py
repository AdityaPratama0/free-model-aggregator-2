"""make.fun free-model aggregator entry, free-model-aggregator-02.

One process, standard library only. It serves GET /v1/models and the two completion wires,
POST /v1/chat/completions and POST /v1/messages, each as JSON or as server-sent events, then
registers its catalog with the trial env and exits 0 when the env reports the trial done.

The brief at $MAKEFUN_PROMPT_FILE names the free OpenRouter models and the prompt manifest. Every
answer is dialled through the gateway at $OPENAI_BASE_URL, one free model per call, and the model id
reported back to the client is always the id that was dialled.

Written as one file on purpose: the trial runs `python3 -P -s /bot/main.py`, so the script directory
is not on sys.path and a sibling module would not import.
"""

import codecs
import json
import math
import os
import signal
import socketserver
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

# --------------------------------------------------------------------------- configuration

PROTOCOL_VERSION = 1
POLL_INTERVAL_SEC = 0.1
MAX_CLIENT_BODY = 8 * 1024 * 1024
GATEWAY_TIMEOUT_SEC = 180.0
MAX_UPSTREAM_ATTEMPTS = 6
MAX_TRAIL_ROWS = 8

COOLDOWN_RATE_LIMITED_SEC = 65.0
COOLDOWN_SERVER_ERROR_SEC = 20.0
COOLDOWN_UNREACHABLE_SEC = 10.0

SCHEMA_PREFIX = 'free-model-aggregator-02/'
MODELS_SCHEMA = 'free-model-aggregator-02/models/v1'
PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1'

FIXED_CREATED = 1767225600
FIXED_CREATED_AT = '2026-01-01T00:00:00Z'

# The public base URL of the provider the free models belong to. It is part of the registered
# catalog and is never dialled by this process; the gateway at $OPENAI_BASE_URL is.
OPENROUTER_PUBLIC_BASE = 'https://' + 'openrouter' + '.ai/api/v1'
OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY'

ENV_URL = os.environ.get('MAKEFUN_ENV_URL', '').rstrip('/')
PROMPT_FILE = os.environ.get('MAKEFUN_PROMPT_FILE', '')
GATEWAY_URL = os.environ.get('OPENAI_BASE_URL', '').rstrip('/')
GATEWAY_KEY = os.environ.get('OPENAI_API_KEY', '')

# Every outbound call goes direct, the way Node's fetch does: the trial's gateway needs no proxy, and
# honouring an inherited http_proxy here would send localhost calls at an unrelated proxy instead.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

Json = dict[str, object]

# --------------------------------------------------------------------------- small type checks


def is_record(value: object) -> bool:
    """A JSON object, never an array or another value."""
    return isinstance(value, dict)


def is_number(value: object) -> bool:
    """A JSON number. Python's bool is an int, so bools are excluded here."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def string_of(value: object) -> str:
    return value if isinstance(value, str) else ''


def number_or_null(value: object) -> int | float | None:
    if is_number(value) and math.isfinite(value):
        return value
    return None


def number_or_zero(value: object) -> int | float:
    return value if is_number(value) else 0


def bool_or_null(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def log(message: str) -> None:
    sys.stderr.write(f'aggregator: {message}\n')
    sys.stderr.flush()


# --------------------------------------------------------------------------- the brief's documents


def object_span(text: str, start: int) -> int:
    """Index of the `}` that closes the object opening at `start`, or -1. String contents are skipped."""
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return index
    return -1


def documents_in(text: str) -> list[Json]:
    """Every JSON object in `text` carrying a string `schema`, in the order it appears."""
    found: list[Json] = []
    start = text.find('{')
    while start != -1:
        end = object_span(text, start)
        if end != -1:
            try:
                value = json.loads(text[start : end + 1])
            except (json.JSONDecodeError, ValueError):
                value = None
            if is_record(value) and isinstance(value.get('schema'), str):
                found.append(value)
        start = text.find('{', start + 1)
    return found


def find_array_document(docs: list[Json], keys: tuple[str, ...]) -> tuple[Json, str] | None:
    """First document under this contest's schema prefix that holds one of `keys` as an array."""
    for doc in docs:
        if SCHEMA_PREFIX not in string_of(doc.get('schema')):
            continue
        for key in keys:
            rows = doc.get(key)
            if isinstance(rows, list):
                return doc, key
    return None


def model_of(value: object) -> Json | None:
    """One brief model row, normalized to the gateway's id spelling and a flat capability record."""
    if not is_record(value):
        return None
    raw = string_of(value.get('id')) or string_of(value.get('model'))
    if raw == '':
        return None
    provider = string_of(value.get('provider')) or string_of(value.get('wire')) or 'openrouter'
    if provider == 'openrouter' and not raw.lower().startswith('openrouter/'):
        model_id = f'openrouter/{raw}'
    else:
        model_id = raw
    display_name = string_of(value.get('display_name')) or raw
    rating = number_or_null(value.get('rating_milli'))
    return {
        'id': model_id,
        'provider': provider,
        'display_name': display_name,
        'rating_milli': rating if rating is not None else 0,
        'context_length': number_or_null(value.get('context_length')),
        'tools': bool_or_null(value.get('tools')),
        'vision': bool_or_null(value.get('vision')),
        'json': bool_or_null(value.get('json')),
        'free': value.get('free') is not False,
    }


def normalize_requires(value: Json) -> Json:
    """The capability flags a prompt actually requires, with everything else dropped."""
    requires: Json = {}
    if value.get('tools') is True:
        requires['tools'] = True
    if value.get('vision') is True:
        requires['vision'] = True
    if value.get('json') is True:
        requires['json'] = True
    min_context = number_or_null(value.get('min_context'))
    if min_context is not None:
        requires['min_context'] = min_context
    return requires


def prompt_of(value: object) -> Json | None:
    """One brief prompt row: wire, stream flag, weight, requirements and answer token."""
    if not is_record(value):
        return None
    prompt_id = string_of(value.get('id'))
    if prompt_id == '':
        return None
    merged: Json = {}
    nested = value.get('requires')
    if is_record(nested):
        merged.update(nested)
    for key in ('tools', 'vision', 'json'):
        if value.get(key) is True:
            merged[key] = True
    if 'min_context' not in merged and is_number(value.get('min_context')):
        merged['min_context'] = value['min_context']
    return {
        'id': prompt_id,
        'api': 'anthropic' if value.get('api') == 'anthropic' else 'openai',
        'stream': value.get('stream') is True,
        'weight': 2 if value.get('weight') == 2 else 1,
        'requires': normalize_requires(merged),
    }


def read_brief(path: str) -> Json:
    """The brief's model list and prompt manifest, found by schema, never by parsing the prose."""
    empty: Json = {'models': [], 'prompts': [], 'error': None}
    if path == '':
        empty['error'] = 'MAKEFUN_PROMPT_FILE is not set'
        return empty
    try:
        with open(path, encoding='utf-8') as handle:
            text = handle.read()
    except OSError as err:
        empty['error'] = f'cannot read {path}: {err}'
        return empty
    docs = documents_in(text)
    models_doc = find_array_document(docs, ('models', 'ratings'))
    if models_doc is None:
        empty['error'] = f'the brief at {path} holds no {MODELS_SCHEMA} document'
        return empty
    models = [m for m in (model_of(row) for row in models_doc[0][models_doc[1]]) if m is not None]
    brief: Json = {'models': models, 'prompts': [], 'error': None}
    prompts_doc = find_array_document(docs, ('prompts', 'manifest'))
    if prompts_doc is None:
        brief['error'] = f'the brief at {path} holds no {PROMPTS_SCHEMA} document'
        return brief
    brief['prompts'] = [p for p in (prompt_of(row) for row in prompts_doc[0][prompts_doc[1]]) if p is not None]
    return brief


def free_models(brief: Json) -> list[Json]:
    """The rows this entry may serve: the brief's free models, in the brief's order."""
    rows = brief.get('models')
    if not isinstance(rows, list):
        return []
    return [row for row in rows if is_record(row) and row.get('free')]


def build_catalog(brief: Json) -> Json:
    """The register body's catalog: one provider, exactly the free model ids the brief names."""
    seen: set[str] = set()
    models: list[Json] = []
    for row in free_models(brief):
        model_id = string_of(row.get('id'))
        if model_id == '' or model_id in seen:
            continue
        seen.add(model_id)
        models.append({'id': model_id, 'free': True, 'context_length': number_or_null(row.get('context_length'))})
    if not models:
        return {'providers': []}
    return {
        'providers': [
            {
                'id': 'openrouter',
                'base_url': OPENROUTER_PUBLIC_BASE,
                'key_env': OPENROUTER_KEY_ENV,
                'wire': 'openai',
                'models': models,
            }
        ]
    }


# --------------------------------------------------------------------------- client wires


def text_of(content: object) -> str:
    """The text of a string, or of a block array's text blocks, joined; '' when there is none."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ''
    parts = [b['text'] for b in content if is_record(b) and b.get('type') == 'text' and isinstance(b.get('text'), str)]
    return '\n'.join(parts)


def content_text(value: object) -> str:
    """An OpenAI message's content as text: a plain string, or the text of its parts."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return text_of(value)
    return ''


def image_part(block: Json) -> Json | None:
    """One Anthropic image block as an OpenAI image_url part; None for anything else."""
    source = block.get('source')
    if not is_record(source):
        return None
    if source.get('type') == 'base64' and isinstance(source.get('data'), str):
        media = source.get('media_type') if isinstance(source.get('media_type'), str) else 'image/png'
        return {'type': 'image_url', 'image_url': {'url': f'data:{media};base64,{source["data"]}'}}
    if source.get('type') == 'url' and isinstance(source.get('url'), str):
        return {'type': 'image_url', 'image_url': {'url': source['url']}}
    return None


def content_part(block: Json) -> Json | None:
    """One Anthropic content block as an OpenAI content part: text or image; None for the rest."""
    kind = block.get('type')
    if kind == 'text' and isinstance(block.get('text'), str):
        return {'type': 'text', 'text': block['text']}
    if kind == 'image':
        return image_part(block)
    return None


def result_text(content: object) -> str:
    """A tool_result block's content as text: its text blocks, else its JSON."""
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of(content)
        if text != '':
            return text
    return json.dumps(content, separators=(',', ':'), ensure_ascii=False)


def arguments_object(value: object) -> Json:
    """A tool call's arguments as an object: an already parsed object, a JSON string, or {}."""
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != '':
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return {}
        if is_record(parsed):
            return parsed
    return {}


def openai_tool(tool: Json) -> Json | None:
    """One Anthropic tool definition as an OpenAI function tool; None when it has no usable name."""
    name = tool.get('name')
    if not isinstance(name, str) or name == '':
        return None
    parameters = tool.get('input_schema')
    return {
        'type': 'function',
        'function': {
            'name': name,
            'description': tool['description'] if isinstance(tool.get('description'), str) else '',
            'parameters': parameters if is_record(parameters) else {'type': 'object', 'properties': {}},
        },
    }


def tool_choice_of(choice: object) -> object | None:
    """An Anthropic tool_choice as OpenAI's; None when there is nothing to pass on."""
    if not is_record(choice):
        return None
    kind = choice.get('type')
    if kind == 'auto':
        return 'auto'
    if kind == 'any':
        return 'required'
    if kind == 'none':
        return 'none'
    if kind == 'tool' and isinstance(choice.get('name'), str):
        return {'type': 'function', 'function': {'name': choice['name']}}
    return None


def assistant_turn(blocks: list, text: str) -> Json:
    """One Anthropic assistant message as an OpenAI assistant message, tool_use blocks as tool_calls."""
    calls: list[Json] = []
    for block in blocks:
        if not is_record(block) or block.get('type') != 'tool_use':
            continue
        name = block.get('name')
        if not isinstance(name, str):
            continue
        calls.append(
            {
                'id': block['id'] if isinstance(block.get('id'), str) else f'toolu_{len(calls)}',
                'type': 'function',
                'function': {
                    'name': name,
                    'arguments': json.dumps(arguments_object(block.get('input')), separators=(',', ':'), ensure_ascii=False),
                },
            }
        )
    message: Json = {'role': 'assistant', 'content': text if text != '' else None}
    if calls:
        message['tool_calls'] = calls
    return message


def anthropic_to_openai(body: Json, model: str) -> Json:
    """An Anthropic Messages request as the Chat Completions request that names `model`.

    System blocks become one system message, tool_use blocks become tool_calls, tool_result blocks
    become tool messages, image blocks become image_url data URLs. Parameters the two wires share
    (max_tokens, temperature, top_p, stop sequences, tools, tool_choice) pass through, and a streaming
    request asks the gateway for a usage chunk so the Anthropic message_delta can carry token counts.
    """
    messages: list[Json] = []
    system = text_of(body.get('system'))
    if system != '':
        messages.append({'role': 'system', 'content': system})
    source = body.get('messages') if isinstance(body.get('messages'), list) else []
    for raw in source:
        if not is_record(raw):
            continue
        assistant = raw.get('role') == 'assistant'
        content = raw.get('content')
        if isinstance(content, str):
            messages.append({'role': 'assistant' if assistant else 'user', 'content': content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            messages.append(assistant_turn(content, text_of(content)))
            continue
        parts: list[Json] = []
        for block in content:
            if not is_record(block):
                continue
            kind = block.get('type')
            if kind == 'tool_result':
                messages.append(
                    {
                        'role': 'tool',
                        'tool_call_id': block['tool_use_id'] if isinstance(block.get('tool_use_id'), str) else '',
                        'content': result_text(block.get('content')),
                    }
                )
                continue
            part = content_part(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({'role': 'user', 'content': parts})

    out: Json = {'model': model, 'messages': messages}
    if is_number(body.get('max_tokens')):
        out['max_tokens'] = body['max_tokens']
    if is_number(body.get('temperature')):
        out['temperature'] = body['temperature']
    if is_number(body.get('top_p')):
        out['top_p'] = body['top_p']
    stops = body.get('stop_sequences')
    if isinstance(stops, list) and stops:
        out['stop'] = stops
    tools = body.get('tools')
    if isinstance(tools, list):
        converted = [t for t in (openai_tool(tool) for tool in tools if is_record(tool)) if t is not None]
        if converted:
            out['tools'] = converted
    choice = tool_choice_of(body.get('tool_choice'))
    if choice is not None:
        out['tool_choice'] = choice
    if body.get('stream') is True:
        out['stream'] = True
        out['stream_options'] = {'include_usage': True}
    return out


def stop_reason_of(finish: object) -> str:
    """OpenAI's finish_reason as Anthropic's stop_reason."""
    if finish == 'length':
        return 'max_tokens'
    if finish == 'tool_calls' or finish == 'function_call':
        return 'tool_use'
    if finish == 'content_filter':
        return 'refusal'
    return 'end_turn'


def openai_to_anthropic(body: Json, model: str) -> Json:
    """A Chat Completions answer as an Anthropic Messages object. `model` is the id actually dialled."""
    choices = body.get('choices') if isinstance(body.get('choices'), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get('message') if is_record(choice.get('message')) else {}
    content: list[Json] = []
    text = content_text(message.get('content'))
    if text != '':
        content.append({'type': 'text', 'text': text})
    calls = message.get('tool_calls') if isinstance(message.get('tool_calls'), list) else []
    for index, call in enumerate(calls):
        if not is_record(call) or not is_record(call.get('function')):
            continue
        fn = call['function']
        content.append(
            {
                'type': 'tool_use',
                'id': call['id'] if isinstance(call.get('id'), str) and call['id'] != '' else f'toolu_{index}',
                'name': fn['name'] if isinstance(fn.get('name'), str) else '',
                'input': arguments_object(fn.get('arguments')),
            }
        )
    if not content:
        content.append({'type': 'text', 'text': ''})
    usage = body.get('usage') if is_record(body.get('usage')) else {}
    return {
        'id': f'msg_{body["id"] if isinstance(body.get("id"), str) and body["id"] != "" else "aggregate"}',
        'type': 'message',
        'role': 'assistant',
        'model': model,
        'content': content,
        'stop_reason': stop_reason_of(choice.get('finish_reason')),
        'stop_sequence': None,
        'usage': {
            'input_tokens': number_or_zero(usage.get('prompt_tokens')),
            'output_tokens': number_or_zero(usage.get('completion_tokens')),
        },
    }


def openai_json_with_model(body: Json, model: str) -> Json:
    """A Chat Completions answer with `model` set to the id the entry dialled."""
    return {**body, 'model': model}


def sse_event(event: str | None, data: object) -> str:
    """One server-sent event: an optional `event:` line, the JSON data line, then the blank line."""
    prefix = '' if event is None else f'event: {event}\n'
    return f'{prefix}data: {json.dumps(data, separators=(",", ":"), ensure_ascii=False)}\n\n'


def frame_with_model(frame: str, model: str) -> str:
    """One SSE frame with each JSON payload's `model` set to `model`; everything else is left alone."""
    lines: list[str] = []
    for line in frame.split('\n'):
        trimmed = line.strip()
        if not trimmed.startswith('data:'):
            lines.append(line)
            continue
        payload = trimmed[5:].strip()
        if payload == '' or payload == '[DONE]':
            lines.append(line)
            continue
        try:
            chunk = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            lines.append(line)
            continue
        if is_record(chunk):
            lines.append(f'data: {json.dumps({**chunk, "model": model}, separators=(",", ":"), ensure_ascii=False)}')
        else:
            lines.append(line)
    return '\n'.join(lines) + '\n\n'


def iter_frames(chunks):
    """SSE frames from an upstream body's byte chunks, split at the blank line that ends each frame.

    A frame cut across reads is held until its end arrives; a body that ends without one is yielded
    as it stands. CRLF is read as the SSE grammar's LF.
    """
    buffer = ''
    decoder = codecs.getincrementaldecoder('utf-8')('replace')
    for chunk in chunks:
        # Combine first, then normalize: a CRLF pair split across two reads still becomes one LF.
        buffer = (buffer + decoder.decode(chunk)).replace('\r\n', '\n')
        at = buffer.find('\n\n')
        while at != -1:
            yield buffer[:at]
            buffer = buffer[at + 2 :]
            at = buffer.find('\n\n')
    buffer = (buffer + decoder.decode(b'', final=True)).replace('\r\n', '\n')
    if buffer.strip() != '':
        yield buffer


def openai_stream_with_model(chunks, model: str):
    """A Chat Completions SSE stream, frame by frame, each payload naming the id the entry dialled."""
    for frame in iter_frames(chunks):
        yield frame_with_model(frame, model)


def usage_of(chunk: Json) -> tuple[int | float, int | float] | None:
    """A chunk's prompt and completion token counts, or None when the chunk carries no usage object."""
    usage = chunk.get('usage')
    if not is_record(usage):
        return None
    return number_or_zero(usage.get('prompt_tokens')), number_or_zero(usage.get('completion_tokens'))


def anthropic_events_from_openai(chunks, model: str):
    """A Chat Completions SSE stream as Anthropic's event sequence, naming `model` back to the client.

    message_start opens the message, each text run or tool call is one content block, and message_delta
    closes it with the stop reason and the token counts the stream reported (or zeros when it reported
    none). The upstream's finish_reason is held until the usage chunk, because Anthropic's message_delta
    is the event that carries both.
    """
    started = False
    finished = False
    pending_stop: str | None = None
    text_index = -1
    next_index = 0
    tool_indexes: dict[int, int] = {}
    input_tokens: int | float = 0
    output_tokens: int | float = 0

    def start_events() -> list[str]:
        nonlocal started
        if started:
            return []
        started = True
        return [
            sse_event(
                'message_start',
                {
                    'type': 'message_start',
                    'message': {
                        'id': 'msg_stream',
                        'type': 'message',
                        'role': 'assistant',
                        'model': model,
                        'content': [],
                        'stop_reason': None,
                        'stop_sequence': None,
                        'usage': {'input_tokens': input_tokens, 'output_tokens': 0},
                    },
                },
            )
        ]

    def close_text() -> list[str]:
        nonlocal text_index
        if text_index == -1:
            return []
        index = text_index
        text_index = -1
        return [sse_event('content_block_stop', {'type': 'content_block_stop', 'index': index})]

    def finish_events(stop_reason: str) -> list[str]:
        nonlocal finished
        if finished:
            return []
        finished = True
        out = close_text()
        for index in sorted(tool_indexes.values()):
            out.append(sse_event('content_block_stop', {'type': 'content_block_stop', 'index': index}))
        tool_indexes.clear()
        out.append(
            sse_event(
                'message_delta',
                {'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}},
            )
        )
        out.append(sse_event('message_stop', {'type': 'message_stop'}))
        return out

    def frame_events(frame: str) -> list[str]:
        nonlocal input_tokens, output_tokens, text_index, next_index, pending_stop
        out: list[str] = []
        for line in frame.split('\n'):
            trimmed = line.strip()
            if not trimmed.startswith('data:'):
                continue
            payload = trimmed[5:].strip()
            if payload == '[DONE]':
                out.extend(start_events())
                out.extend(finish_events(pending_stop if pending_stop is not None else 'end_turn'))
                continue
            try:
                chunk = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                continue
            if not is_record(chunk):
                continue
            usage = usage_of(chunk)
            if usage is not None:
                input_tokens, output_tokens = usage
            choices = chunk.get('choices')
            if not isinstance(choices, list) or not choices or not is_record(choices[0]):
                continue
            choice = choices[0]
            delta = choice.get('delta') if is_record(choice.get('delta')) else {}
            out.extend(start_events())
            text = delta.get('content')
            if isinstance(text, str) and text != '':
                if text_index == -1:
                    text_index = next_index
                    next_index += 1
                    out.append(
                        sse_event(
                            'content_block_start',
                            {'type': 'content_block_start', 'index': text_index, 'content_block': {'type': 'text', 'text': ''}},
                        )
                    )
                out.append(
                    sse_event(
                        'content_block_delta',
                        {'type': 'content_block_delta', 'index': text_index, 'delta': {'type': 'text_delta', 'text': text}},
                    )
                )
            calls = delta.get('tool_calls')
            for position, call in enumerate(calls if isinstance(calls, list) else []):
                if not is_record(call):
                    continue
                upstream_index = int(call['index']) if is_number(call.get('index')) else position
                index = tool_indexes.get(upstream_index)
                fn = call.get('function') if is_record(call.get('function')) else {}
                if index is None:
                    out.extend(close_text())
                    index = next_index
                    next_index += 1
                    tool_indexes[upstream_index] = index
                    out.append(
                        sse_event(
                            'content_block_start',
                            {
                                'type': 'content_block_start',
                                'index': index,
                                'content_block': {
                                    'type': 'tool_use',
                                    'id': call['id'] if isinstance(call.get('id'), str) and call['id'] != '' else f'toolu_{index}',
                                    'name': fn['name'] if isinstance(fn.get('name'), str) else '',
                                    'input': {},
                                },
                            },
                        )
                    )
                if isinstance(fn.get('arguments'), str) and fn['arguments'] != '':
                    out.append(
                        sse_event(
                            'content_block_delta',
                            {'type': 'content_block_delta', 'index': index, 'delta': {'type': 'input_json_delta', 'partial_json': fn['arguments']}},
                        )
                    )
            if choice.get('finish_reason') is not None:
                pending_stop = stop_reason_of(choice.get('finish_reason'))
        return out

    for frame in iter_frames(chunks):
        for event in frame_events(frame):
            yield event
    if not finished:
        for event in start_events():
            yield event
        for event in finish_events(pending_stop if pending_stop is not None else 'end_turn'):
            yield event


# --------------------------------------------------------------------------- the trial state

BRIEF = read_brief(PROMPT_FILE)
CATALOG = build_catalog(BRIEF)

# The brief's prompts arrive in the order it lists them, so the next request is the next row. The
# index only decides which requirements to check; it never changes the answer.
manifest_index = 0
manifest_lock = threading.Lock()

# A model that refused is skipped until its window passes, so one rate-limited model cannot take
# every later prompt down with it. Cooling reorders the candidates; it never drops one.
COOLDOWNS: dict[str, float] = {}
COOLDOWN_LOCK = threading.Lock()


def now_ts() -> float:
    return time.monotonic()


def cooling(model_id: str) -> bool:
    """Whether the model is inside a cooldown window; an expired window is cleared as it is read."""
    with COOLDOWN_LOCK:
        until = COOLDOWNS.get(model_id)
        if until is None:
            return False
        if until <= now_ts():
            del COOLDOWNS[model_id]
            return False
        return True


def mark_cooldown(model_id: str, status: int) -> None:
    """Start a cooldown after a refusal: a rate limit is long, a server error or a dead socket short."""
    if status == 429:
        seconds = COOLDOWN_RATE_LIMITED_SEC
    elif status == 0:
        seconds = COOLDOWN_UNREACHABLE_SEC
    elif status >= 500:
        seconds = COOLDOWN_SERVER_ERROR_SEC
    else:
        return
    with COOLDOWN_LOCK:
        COOLDOWNS[model_id] = now_ts() + seconds


# --------------------------------------------------------------------------- choosing a model


def capabilities_of(model: Json) -> Json:
    return {
        'context_length': model.get('context_length'),
        'tools': model.get('tools'),
        'vision': model.get('vision'),
        'json': model.get('json'),
    }


def meets_requirements(requires: Json, caps: Json) -> bool:
    """Whether a model's capabilities meet a prompt's requirements. An unknown capability meets none."""
    if requires.get('tools') is True and caps.get('tools') is not True:
        return False
    if requires.get('vision') is True and caps.get('vision') is not True:
        return False
    if requires.get('json') is True and caps.get('json') is not True:
        return False
    min_context = requires.get('min_context')
    if min_context is not None:
        context = caps.get('context_length')
        if not is_number(context) or context < min_context:
            return False
    return True


def candidates_for(requires: Json) -> list[Json]:
    """Every free model that meets the requirements, best first, cooling models last.

    Quality is 60% of the score, so the order is rating_milli highest first and the brief's own order
    breaks ties (sorted() is stable). A model cooling after a refusal moves behind every warm one but is
    still returned, because a tired model beats no answer at all.
    """
    eligible = [m for m in free_models(BRIEF) if meets_requirements(requires, capabilities_of(m))]
    warm = [m for m in eligible if not cooling(string_of(m.get('id')))]
    cold = [m for m in eligible if cooling(string_of(m.get('id')))]
    warm.sort(key=lambda m: number_or_zero(m.get('rating_milli')), reverse=True)
    cold.sort(key=lambda m: number_or_zero(m.get('rating_milli')), reverse=True)
    return warm + cold


def route_for(requires: Json) -> Json | None:
    """The best free model that serves a request, or None when none can."""
    ranked = candidates_for(requires)
    return ranked[0] if ranked else None


def requirements_from_body(body: Json) -> Json:
    """What a request body itself demands: tools it passes, images it carries, JSON mode it asks for."""
    messages = body.get('messages') if isinstance(body.get('messages'), list) else []
    vision = False
    for message in messages:
        if not is_record(message):
            continue
        content = message.get('content')
        if not isinstance(content, list):
            continue
        for block in content:
            if is_record(block) and block.get('type') in ('image', 'image_url'):
                vision = True
    tools = body.get('tools')
    response_format = body.get('response_format')
    response_type = response_format.get('type') if is_record(response_format) else None
    requires: Json = {}
    if isinstance(tools, list) and tools:
        requires['tools'] = True
    if vision:
        requires['vision'] = True
    if response_type in ('json_object', 'json_schema'):
        requires['json'] = True
    return requires


def requirements_for(body: Json) -> Json:
    """The requirements the arriving prompt states, from the brief's next row and from the body.

    The brief's row is the declared source; the body is the evidence. Their union is what the model
    must actually handle, so neither one alone can pull in a model that cannot serve the request.
    """
    global manifest_index
    with manifest_lock:
        index = manifest_index
        manifest_index += 1
    declared: Json = {}
    prompts = BRIEF.get('prompts')
    if isinstance(prompts, list) and index < len(prompts):
        row = prompts[index]
        if is_record(row) and is_record(row.get('requires')):
            declared = normalize_requires(row['requires'])
    observed = requirements_from_body(body)
    merged = dict(declared)
    merged.update(observed)
    return merged


def served_models() -> list[Json]:
    """The free models the brief lists, as GET /v1/models names them."""
    out: list[Json] = []
    for row in free_models(BRIEF):
        out.append(
            {
                'id': row['id'],
                'provider': row.get('provider') if isinstance(row.get('provider'), str) else 'openrouter',
                'display_name': row.get('display_name') if isinstance(row.get('display_name'), str) else row['id'],
            }
        )
    return out


def openai_model_list(models: list[Json]) -> Json:
    """GET /v1/models on an OpenAI-shaped request, in OpenAI's list shape."""
    return {
        'object': 'list',
        'data': [
            {'id': m['id'], 'object': 'model', 'created': FIXED_CREATED, 'owned_by': string_of(m.get('provider'))}
            for m in models
        ],
    }


def anthropic_model_list(models: list[Json]) -> Json:
    """GET /v1/models on an Anthropic-shaped request (an x-api-key header), in Anthropic's list shape."""
    return {
        'data': [
            {'type': 'model', 'id': m['id'], 'display_name': string_of(m.get('display_name')), 'created_at': FIXED_CREATED_AT}
            for m in models
        ],
        'has_more': False,
    }


def anthropic_error(error_type: str, message: str) -> Json:
    """An error in Anthropic's shape (the Messages API's error envelope)."""
    return {'type': 'error', 'error': {'type': error_type, 'message': message}}


def openai_error(message: str, error_type: str, code: str | None) -> Json:
    """An error in OpenAI's shape (Chat Completions)."""
    return {'error': {'message': message, 'type': error_type, 'code': code}}


# --------------------------------------------------------------------------- the gateway


class Upstream:
    """One gateway answer: its status, and a body read either whole or as it arrives."""

    def __init__(self, status: int, response) -> None:
        self.status = status
        self._response = response

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def chunks(self):
        """The body's bytes as they arrive, never buffered until full: an SSE stream must stream."""
        reader = getattr(self._response, 'read1', None) or self._response.read
        while True:
            chunk = reader(65536)
            if not chunk:
                return
            yield chunk

    def text(self) -> str:
        try:
            return self._response.read().decode('utf-8', 'replace')
        finally:
            self.close()

    def close(self) -> None:
        try:
            self._response.close()
        except Exception:  # a socket already closed is nothing to report
            pass


def chat_completions_url() -> str:
    """The gateway's Chat Completions route.

    OPENAI_BASE_URL is the OpenAI-compatible root, which already ends in `/v1` on the trial network; a
    base that does not gets one, so both spellings reach the same route.
    """
    if GATEWAY_URL == '':
        return ''
    return f'{GATEWAY_URL}/chat/completions' if GATEWAY_URL.endswith('/v1') else f'{GATEWAY_URL}/v1/chat/completions'


def dial(model: str, body: Json) -> Upstream:
    """POST one request to the gateway, naming `model`: the id the entry claims it dialled.

    A refusal is still an answer, with a status and a body, exactly as fetch reports it. Nothing is
    retried here; retrying is the caller's job, on the next candidate, so each gateway call is counted.
    """
    url = chat_completions_url()
    if url == '':
        raise RuntimeError('OPENAI_BASE_URL is not set')
    data = json.dumps({**body, 'model': model}, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    headers = {'content-type': 'application/json', 'authorization': f'Bearer {GATEWAY_KEY}'}
    if body.get('stream') is True:
        headers['accept'] = 'text/event-stream'
    request = urllib.request.Request(url, data=data, headers=headers, method='POST')
    try:
        response = OPENER.open(request, timeout=GATEWAY_TIMEOUT_SEC)
    except urllib.error.HTTPError as err:
        return Upstream(err.code, err)
    return Upstream(getattr(response, 'status', 200), response)


# --------------------------------------------------------------------------- the HTTP server


def requirements_or_all(body: Json) -> Json:
    """The requirements to route by: the declared union first, then the body's alone, then nothing.

    A brief row whose declared requirements exclude every model must not cost the prompt. Each fallback
    is strictly wider, so the entry always tries something, and the trail still records what refused.
    """
    merged = requirements_for(body)
    if candidates_for(merged):
        return merged
    observed = requirements_from_body(body)
    if candidates_for(observed):
        return observed
    return {}


class Server(ThreadingHTTPServer):
    """The listening socket, with HTTPServer's reverse-DNS lookup removed.

    HTTPServer.server_bind calls socket.getfqdn(host) to fill server_name. On a host whose resolver is
    slow or absent that call blocks for many seconds before the port is even known, which stalls
    registration and the whole trial. The name is never read here, so it is skipped.
    """

    daemon_threads = True
    allow_reuse_address = True

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        self.server_name = 'localhost'
        self.server_port = self.server_address[1]


class Handler(BaseHTTPRequestHandler):
    """The entry's one server: the models route, the two completion wires, nothing else."""

    protocol_version = 'HTTP/1.1'
    server_version = 'makefun-aggregator-py/2'
    sys_version = ''

    def log_message(self, format: str, *args: object) -> None:  # the base class's name
        """The trial's stderr is this entry's own log, so no access line is written per request."""

    def do_GET(self) -> None:  # the base class's name
        self.dispatch()

    def do_POST(self) -> None:  # the base class's name
        self.dispatch()

    # -- plumbing -----------------------------------------------------------------------------------

    def dispatch(self) -> None:
        self.headers_sent = False
        path = urlsplit(self.path).path
        try:
            if self.command == 'GET' and path == '/v1/models':
                self.models()
                return
            if self.command == 'POST' and path == '/v1/chat/completions':
                self.chat()
                return
            if self.command == 'POST' and path == '/v1/messages':
                self.messages()
                return
            wire = 'anthropic' if path.startswith('/v1/messages') else 'openai'
            # A POST to an unknown route leaves its body unread, so keep-alive would read those bytes
            # as the next request line. Close instead.
            self.close_connection = True
            self.send_error_body(wire, 404, 'not_found', f'no route {self.command} {path}')
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            log(traceback.format_exc())
            if self.headers_sent:
                # Mid-answer, the length is already promised: a close is the only honest end.
                self.close_connection = True
                return
            wire = 'anthropic' if urlsplit(self.path).path.startswith('/v1/messages') else 'openai'
            try:
                self.send_error_body(wire, 500, 'internal_error', 'the server failed to answer')
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

    def send_json(self, status: int, body: object, extra: dict[str, str] | None = None) -> None:
        raw = json.dumps(body, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('content-type', 'application/json')
        self.send_header('content-length', str(len(raw)))
        for name, value in (extra or {}).items():
            self.send_header(name, value)
        self.headers_sent = True
        self.end_headers()
        self.wfile.write(raw)

    def send_error_body(self, wire: str, status: int, error_type: str, message: str) -> None:
        """One error in the client's wire, so a client of either wire reads its own envelope."""
        if wire == 'anthropic':
            self.send_json(status, anthropic_error(error_type, message))
        else:
            self.send_json(status, openai_error(message, error_type, None))

    def refused(self, wire: str, status: int, text: str, rows: list[Json]) -> None:
        """A gateway refusal as a 502 in the client's wire, reached only when every candidate failed.

        The failure trail rides on the error too: a client that got no answer still learns which models
        refused and why, rather than only that the aggregate gave up.
        """
        detail = '' if text == '' else f': {text[:500]}'
        if status == 0:
            message = 'no free model could be dialled'
        else:
            message = f'the gateway answered HTTP {status}{detail}'
        extra: dict[str, str] = {}
        if rows:
            extra['x-makefun-upstream-failures'] = json.dumps(rows[:MAX_TRAIL_ROWS], separators=(',', ':'))
        body = anthropic_error('api_error', message) if wire == 'anthropic' else openai_error(message, 'upstream_error', None)
        self.send_json(502, body, extra)

    def stream_headers(self, extra: dict[str, str] | None = None) -> None:
        """Open an SSE answer. http.server does not chunk a length-less response, so the close ends it."""
        self.send_response(200)
        self.send_header('content-type', 'text/event-stream')
        self.send_header('cache-control', 'no-cache')
        self.send_header('x-accel-buffering', 'no')
        for name, value in (extra or {}).items():
            self.send_header(name, value)
        self.send_header('connection', 'close')
        self.headers_sent = True
        self.end_headers()
        self.close_connection = True

    def write_event(self, text: str) -> None:
        self.wfile.write(text.encode('utf-8'))
        self.wfile.flush()

    def read_json(self) -> Json | None:
        """One client request body, or None when it is not one JSON object or is over the size cap."""
        length = self.headers.get('content-length')
        if length is None:
            self.close_connection = True
            return None
        try:
            size = int(length)
        except ValueError:
            self.close_connection = True
            return None
        if size < 0 or size > MAX_CLIENT_BODY:
            self.close_connection = True
            return None
        raw = self.rfile.read(size) if size > 0 else b''
        if not raw:
            return None
        try:
            value = json.loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None
        return value if is_record(value) else None

    # -- the routes ---------------------------------------------------------------------------------

    def models(self) -> None:
        """GET /v1/models, in the shape of the auth header: x-api-key is Anthropic, else OpenAI."""
        listing = served_models()
        if self.headers.get('x-api-key') is not None:
            self.send_json(200, anthropic_model_list(listing))
        else:
            self.send_json(200, openai_model_list(listing))

    def open_upstream(self, wire: str, body: Json, ranked: list[Json]):
        """Dial the ranked candidates in order until one answers, and report each refusal.

        Returns (route, upstream, extra) on the first 2xx. The trail in x-makefun-upstream-failures is
        one row per refused call, in call order, each with exactly `status`, `code` and `model`. A
        refusal starts that model's cooldown; a socket error is a refusal too, with status 0.

        A 4xx other than 429 is not retried across models: it is the request's fault, and every other
        model would refuse it the same way. The client gets that error at once, because spending the
        whole candidate list on a malformed request only raises the call count.
        """
        rows: list[Json] = []
        last_status = 0
        last_text = ''
        for candidate in ranked[:MAX_UPSTREAM_ATTEMPTS]:
            model_id = string_of(candidate.get('id'))
            payload = body if wire == 'openai' else anthropic_to_openai(body, model_id)
            try:
                upstream = dial(model_id, payload)
            except Exception:
                rows.append({'status': 0, 'code': 'unreachable', 'model': model_id})
                mark_cooldown(model_id, 0)
                last_status = 0
                continue
            if upstream.ok:
                extra: dict[str, str] = {}
                if rows:
                    extra['x-makefun-upstream-failures'] = json.dumps(rows[:MAX_TRAIL_ROWS], separators=(',', ':'))
                return candidate, upstream, extra
            text = upstream.text()
            code = ''
            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                parsed = None
            if is_record(parsed):
                error = parsed.get('error')
                if is_record(error):
                    code = string_of(error.get('code'))[:40]
            mark_cooldown(model_id, upstream.status)
            rows.append({'status': upstream.status, 'code': code, 'model': model_id})
            last_status = upstream.status
            last_text = text
            if upstream.status != 429 and 400 <= upstream.status < 500:
                break
        if rows:
            log(f'the trail after {len(rows)} refusal(s): {json.dumps(rows[:MAX_TRAIL_ROWS], separators=(",", ":"))}')
        self.refused(wire, last_status, last_text, rows)
        return None

    def chat(self) -> None:
        """POST /v1/chat/completions: an OpenAI-shaped client, answered by the best model that serves it."""
        body = self.read_json()
        if body is None:
            self.send_error_body('openai', 400, 'invalid_request_error', 'the request body is not one JSON object')
            return
        ranked = candidates_for(requirements_or_all(body))
        if not ranked:
            self.send_error_body('openai', 503, 'no_free_model', 'no free model in the brief meets this request')
            return
        trail = self.open_upstream('openai', body, ranked)
        if trail is None:
            return
        route, upstream, extra = trail
        if body.get('stream') is True:
            self.stream_headers(extra)
            try:
                for frame in openai_stream_with_model(upstream.chunks(), string_of(route.get('id'))):
                    self.write_event(frame)
            finally:
                upstream.close()
            return
        answer = upstream.text()
        try:
            parsed = json.loads(answer)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if not is_record(parsed):
            self.send_error_body('openai', 502, 'upstream_error', 'the gateway answered with something other than JSON')
            return
        self.send_json(200, openai_json_with_model(parsed, string_of(route.get('id'))), extra)

    def messages(self) -> None:
        """POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and back."""
        body = self.read_json()
        if body is None:
            self.send_error_body('anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object')
            return
        ranked = candidates_for(requirements_or_all(body))
        if not ranked:
            self.send_error_body('anthropic', 503, 'overloaded_error', 'no free model in the brief meets this request')
            return
        trail = self.open_upstream('anthropic', body, ranked)
        if trail is None:
            return
        route, upstream, extra = trail
        model_id = string_of(route.get('id'))
        if body.get('stream') is True:
            self.stream_headers(extra)
            try:
                for event in anthropic_events_from_openai(upstream.chunks(), model_id):
                    self.write_event(event)
            finally:
                upstream.close()
            return
        answer = upstream.text()
        try:
            parsed = json.loads(answer)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if not is_record(parsed):
            self.send_error_body('anthropic', 502, 'api_error', 'the gateway answered with something other than JSON')
            return
        self.send_json(200, openai_to_anthropic(parsed, model_id), extra)


# --------------------------------------------------------------------------- register and poll


def register(port: int, catalog: Json) -> None:
    """POST /register with the port and the catalog; the env reports the trial's prompt count back."""
    body = json.dumps({'protocol_version': PROTOCOL_VERSION, 'port': port, 'catalog': catalog}, separators=(',', ':'), ensure_ascii=False)
    request = urllib.request.Request(
        f'{ENV_URL}/register',
        data=body.encode('utf-8'),
        headers={'content-type': 'application/json'},
        method='POST',
    )
    try:
        with OPENER.open(request, timeout=5) as response:
            status = response.status
            reply_text = response.read().decode('utf-8', 'replace')
    except urllib.error.HTTPError as err:
        raise RuntimeError(f'POST {ENV_URL}/register answered HTTP {err.code}: {err.read().decode("utf-8", "replace")[:500]}') from err
    if not (200 <= status < 300):
        raise RuntimeError(f'POST {ENV_URL}/register answered HTTP {status}: {reply_text[:500]}')
    try:
        reply = json.loads(reply_text)
    except (json.JSONDecodeError, ValueError):
        reply = None
    if is_record(reply) and is_number(reply.get('requests')):
        log(f'registered: the trial holds {reply["requests"]} prompts')


def poll() -> None:
    """Poll GET /work every 100 ms and exit 0 the moment it says done."""
    while True:
        time.sleep(POLL_INTERVAL_SEC)
        url = f'{ENV_URL}/work'
        try:
            with OPENER.open(url, timeout=2) as response:
                if response.status != 200:
                    continue
                work = json.loads(response.read().decode('utf-8', 'replace'))
        except Exception:
            continue
        if is_record(work) and work.get('done') is True:
            log('the env says done: exiting 0')
            sys.stderr.flush()
            sys.stdout.flush()
            os._exit(0)


# --------------------------------------------------------------------------- start


def shutdown(*_: object) -> None:
    """SIGTERM and SIGINT end the trial the way Node's process.exit(0) does."""
    sys.stderr.flush()
    os._exit(0)


def start() -> None:
    if BRIEF.get('error') is not None:
        log(string_of(BRIEF.get('error')))
    if '--list' in sys.argv[1:]:
        sys.stdout.write(json.dumps(CATALOG, separators=(',', ':'), ensure_ascii=False) + '\n')
        sys.stdout.flush()
        return
    if ENV_URL == '':
        raise RuntimeError('MAKEFUN_ENV_URL is not set')
    if chat_completions_url() == '':
        raise RuntimeError('OPENAI_BASE_URL is not set')
    server = Server(('0.0.0.0', 0), Handler)
    try:
        port = server.server_address[1]
        if port == 8080:
            raise RuntimeError('the OS picked 8080, which the env owns on one layout: restart the trial')
        log(f'listening on 0.0.0.0:{port} with {len(served_models())} free models, {len(BRIEF.get("prompts") or [])} prompts')
        register(port, CATALOG)
        signal.signal(signal.SIGTERM, shutdown)
        signal.signal(signal.SIGINT, shutdown)
        threading.Thread(target=poll, name='work-poll', daemon=True).start()
        server.serve_forever()
    finally:
        server.server_close()


def main() -> int:
    try:
        start()
        return 0
    except SystemExit:
        raise
    except BaseException:
        log(traceback.format_exc())
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
