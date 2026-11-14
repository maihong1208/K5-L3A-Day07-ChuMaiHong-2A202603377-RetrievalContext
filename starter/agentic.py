"""Example agent loop; connect a real model and the Day 6 retriever via adapters.

Adapters are intentionally not provided: Day 6 has no fixed database schema and
model providers differ. Never use fake adapters to report benchmark quality.
"""
import json
import os
from pathlib import Path
from typing import Any, Protocol

from baseline import pack_context


def load_dotenv() -> None:
    """Load a local .env from the project root when present.

    This keeps API keys out of shell history and allows a project-local secret
    file without requiring the user to export variables manually every run.
    """
    env_paths = [
        Path.cwd() / '.env',
        Path(__file__).resolve().parents[1] / '.env',
        Path(__file__).resolve().parent / '.env',
    ]
    for env_path in env_paths:
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding='utf-8-sig').splitlines():
            line = raw_line.strip().lstrip('\ufeff')
            if not line or line.startswith('#') or '=' not in line:
                continue
            if line.startswith('export '):
                line = line[len('export '):]
            key, value = [part.strip() for part in line.split('=', 1)]
            if not key:
                continue
            key = key.strip()
            value = value.strip().strip("\"'")
            os.environ.setdefault(key, value)
            if key == 'MODEL' and 'OPENAI_MODEL' not in os.environ:
                os.environ['OPENAI_MODEL'] = value
            if key == 'PROVIDER' and value.lower() == 'openai' and 'OPENAI_API_KEY' in os.environ:
                os.environ.setdefault('OPENAI_PROVIDER', value)
        break


load_dotenv()

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional at runtime
    OpenAI = None

try:
    from day6_real_connector import Day6Retriever
except Exception:  # pragma: no cover - optional connector for real Day 6 data
    Day6Retriever = None


class RealDay6Retriever:
    """Default Day 7 retriever wrapper around the actual Day 6 vector store.

    This keeps the Day 6 implementation untouched and exposes the `Retriever.search`
    contract expected by the agent loop.
    """

    def __init__(self, *, strategy: str = "B", model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device: str = "cpu") -> None:
        if Day6Retriever is None:
            raise RuntimeError(
                "Day 6 connector is unavailable. Ensure starter/day6_real_connector.py exists "
                "and the Day 6 project is installed with its dependencies."
            )
        self._inner = Day6Retriever(strategy=strategy, model_name=model_name, device=device)

    def search(self, query: str, *, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        return self._inner.search(query, filters=filters, top_k=top_k)


class Controller(Protocol):
    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Call a real model with a search tool; return normalized action.

        Search: {'action': 'search', 'query': '...'}.
        Final: {'action': 'answer', 'answer': '...', 'citations': ['chunk-id']}.
        Citations must also appear as [chunk-id] in the answer text.
        """
        ...


class OpenAIModelController:
    """OpenAI-compatible controller with env-based configuration.

    Use OPENAI_API_KEY and optionally OPENAI_BASE_URL / OPENAI_MODEL.
    When the environment is not configured, it raises a clear runtime error so
    lab traces can record a provider error instead of silently faking outputs.
    """

    def __init__(self, *, model: str | None = None, api_key: str | None = None, base_url: str | None = None) -> None:
        if OpenAI is None:
            raise RuntimeError('openai package is not installed')
        self.model = model or os.getenv('OPENAI_MODEL') or 'gpt-4o-mini'
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.base_url = base_url or os.getenv('OPENAI_BASE_URL')
        if not self.api_key:
            raise RuntimeError('OPENAI_API_KEY is not configured. Set it before running the agent on a real model.')
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.usage = {'requests': 0, 'input_tokens': 0, 'cached_input_tokens': 0, 'output_tokens': 0}

    def _record_usage(self, response: Any) -> None:
        usage = getattr(response, 'usage', None)
        if usage is None:
            return
        if isinstance(usage, dict):
            input_tokens = usage.get('input_tokens', usage.get('prompt_tokens', 0))
            output_tokens = usage.get('output_tokens', usage.get('completion_tokens', 0))
            details = usage.get('input_tokens_details') or usage.get('prompt_tokens_details') or {}
            cached_input_tokens = details.get('cached_tokens', 0) if isinstance(details, dict) else 0
        else:
            input_tokens = getattr(usage, 'input_tokens', getattr(usage, 'prompt_tokens', 0))
            output_tokens = getattr(usage, 'output_tokens', getattr(usage, 'completion_tokens', 0))
            details = getattr(usage, 'input_tokens_details', getattr(usage, 'prompt_tokens_details', None))
            cached_input_tokens = getattr(details, 'cached_tokens', 0)
        if isinstance(input_tokens, int) and isinstance(output_tokens, int):
            self.usage['requests'] += 1
            self.usage['input_tokens'] += input_tokens
            self.usage['cached_input_tokens'] += cached_input_tokens if isinstance(cached_input_tokens, int) else 0
            self.usage['output_tokens'] += output_tokens

    def _format_history(self, history: list[dict[str, Any]]) -> str:
        if not history:
            return 'No search has been made yet.'
        lines: list[str] = []
        for idx, entry in enumerate(history, 1):
            queries = entry.get('queries') or []
            context = entry.get('context') or ''
            context_ids = entry.get('context_ids') or []
            warning = entry.get('warning') or ''
            lines.append(f"Round {idx}: queries={queries}; context_ids={context_ids}; warning={warning}; context={context}")
        return '\n'.join(lines)

    def _extract_text(self, response: Any) -> str:
        if hasattr(response, 'output_text') and response.output_text:
            return str(response.output_text)
        if hasattr(response, 'choices'):
            msg = response.choices[0].message
            if hasattr(msg, 'content'):
                content = msg.content
                if isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, dict):
                            if block.get('type') == 'text':
                                parts.append(str(block.get('text', '')))
                            elif 'text' in block:
                                parts.append(str(block['text']))
                    return '\n'.join(parts)
                return str(content)
        if isinstance(response, dict):
            choices = response.get('choices') or []
            if choices:
                message = choices[0].get('message') or {}
                content = message.get('content') or ''
                if isinstance(content, list):
                    return '\n'.join(str(part.get('text', '')) for part in content if isinstance(part, dict))
                return str(content)
        return str(response)

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = f"""Bạn là một agent trả lời có căn cứ trên kho dữ liệu tài liệu.
Trả về JSON hợp lệ với đúng 1 trong 2 dạng:
1) {{"action": "search", "query": "..."}}
2) {{"action": "answer", "answer": "... [chunk-id] ...", "citations": ["chunk-id"]}}

Yêu cầu:
- Nếu chưa đủ information, hãy chọn action=search và query ngắn, cụ thể, dùng từ khóa quan trọng từ câu hỏi.
- Nếu đã có đủ evidence trong context, trả lời ngắn gọn, có citation bằng ID chunk trong câu trả lời.
- Chỉ dùng ID trong context_ids, chép nguyên văn; không tự tạo hoặc suy đoán citation ID.
- Nếu câu hỏi yêu cầu con số, ngưỡng hoặc thuật toán cụ thể mà context không nêu rõ, không thay bằng một khuyến nghị chung; hãy tiếp tục tìm hoặc từ chối.
- Nếu truy vấn đã có trong lịch sử, không lặp lại; hãy đổi truy vấn cụ thể hơn hoặc trả lời/từ chối theo context hiện có.
- Nếu không có bằng chứng đủ, không trả lời chắc chắn; hãy dùng action=search thay vì answer.

Câu hỏi: {question}
Lịch sử tìm kiếm:
{self._format_history(history)}
"""
        if hasattr(self.client, 'responses'):
            response = self.client.responses.create(model=self.model, input=prompt, temperature=0)
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0,
            )
        self._record_usage(response)
        text = self._extract_text(response).strip()
        cleaned = text[3:] if text.startswith('```') else text
        cleaned = cleaned[:-3] if cleaned.endswith('```') else cleaned
        try:
            data = json.loads(cleaned.strip())
        except json.JSONDecodeError:
            start = cleaned.find('{')
            end = cleaned.rfind('}')
            if start == -1 or end == -1 or end <= start:
                raise
            data = json.loads(cleaned[start:end + 1])
        action = data.get('action')
        if action == 'search':
            query = data.get('query')
            if not isinstance(query, str) or not query.strip():
                raise ValueError('Model returned invalid search query')
            return {'action': 'search', 'query': query}
        if action == 'answer':
            answer = data.get('answer')
            citations = data.get('citations')
            if not isinstance(answer, str) or not isinstance(citations, list):
                raise ValueError('Model returned invalid answer payload')
            return {'action': 'answer', 'answer': answer, 'citations': citations}
        raise ValueError(f'Unsupported action from model: {action!r}')


class Retriever(Protocol):
    def search(self, query: str, *, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        """Query the real Day 6 vector store with a compatible query embedding.

        Return each chunk's id, text, source_id, source, page, metadata and score.
        The query embedding model and dimension must match the document index.
        """
        ...


def run_agent(question: str, controller: Controller, retriever: Retriever, *,
              filters: dict[str, Any] | None = None, max_searches: int = 4,
              top_k: int = 3, budget: int = 800) -> dict[str, Any]:
    """Return answer, citation IDs and trace; no benchmark gold enters this loop.

    A final answer without valid citations, an empty context, and a search limit
    lead to abstention. This only checks citation IDs, not semantic grounding.
    Provider/retriever errors propagate; no silent mock fallback.
    """
    if not question.strip() or max_searches < 1 or top_k < 1 or budget < 1:
        raise ValueError('question, max_searches, top_k hoặc budget không hợp lệ')
    history: list[dict[str, Any]] = []
    chunks: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    searched_queries: set[str] = set()
    repeated_queries: set[str] = set()
    for _ in range((max_searches * 2) + 1):
        action = controller.decide(question, history)
        if action.get('action') == 'search':
            if sum(e['event'] == 'search' for e in events) >= max_searches:
                return {'status': 'abstain', 'answer': 'Không đủ bằng chứng để trả lời.', 'citations': [], 'events': events + [{'event': 'search_limit'}]}
            query = action.get('query')
            if not isinstance(query, str) or not query.strip():
                raise ValueError('search query phải là chuỗi không rỗng')
            normalized_query = ' '.join(query.casefold().split())
            fallback = False
            if normalized_query in searched_queries:
                if normalized_query in repeated_queries:
                    return {'status': 'abstain', 'answer': 'Không đủ bằng chứng để trả lời.', 'citations': [],
                            'events': events + [{'event': 'repeated_search_limit', 'query': query}]}
                repeated_queries.add(normalized_query)
                events.append({'event': 'repeated_search', 'query': query})
                history[-1]['warning'] = (
                    f'The query "{query}" was already searched and produced no new evidence. '
                    'Do not repeat it; answer from context or abstain. A fallback search with the original question will be tried.'
                )
                query = question.strip()
                normalized_query = ' '.join(query.casefold().split())
                fallback = True
                if normalized_query in searched_queries:
                    continue
            searched_queries.add(normalized_query)
            results = retriever.search(query, filters=filters, top_k=top_k)
            if len(results) > top_k or len({r['id'] for r in results}) != len(results):
                raise ValueError('retriever phải trả tối đa top_k chunk ID duy nhất')
            candidates.extend(results)
            context, ids = pack_context(candidates, budget=budget)
            chunks = {c['id']: c for c in candidates if c['id'] in ids}
            events.append({'event': 'search', 'query': query, 'filter': filters, 'results': results,
                           'context': context, 'context_ids': ids, 'fallback': fallback})
            # Send only the latest cumulative context; avoid multiplying the budget.
            history = [{'tool': 'search', 'queries': [e['query'] for e in events if e['event'] == 'search'],
                        'context': context, 'context_ids': ids}]
        elif action.get('action') == 'answer':
            answer, citations = action.get('answer'), action.get('citations')
            if not isinstance(answer, str) or not isinstance(citations, list) or any(not isinstance(i, str) for i in citations):
                raise ValueError('answer/citations không hợp lệ')
            valid = bool(answer.strip() and citations and all(i in chunks for i in citations))
            if valid:
                missing_inline = [cid for cid in citations if f'[{cid}]' not in answer]
                if missing_inline:
                    answer = f"{answer.rstrip()} {' '.join(f'[{cid}]' for cid in missing_inline)}"
                return {'status': 'answered', 'answer': answer, 'citations': citations,
                        'events': events + [{'event': 'answer', 'citation_ids_appended': missing_inline}]}
            return {'status': 'abstain', 'answer': 'Không đủ bằng chứng để trả lời.', 'citations': [],
                    'events': events + [{'event': 'invalid_or_missing_citations',
                                         'candidate_answer': answer, 'candidate_citations': citations}]}
        else:
            raise ValueError('action phải là search hoặc answer')
    return {'status': 'abstain', 'answer': 'Không đủ bằng chứng để trả lời.', 'citations': [],
            'events': events + [{'event': 'decision_limit'}]}
