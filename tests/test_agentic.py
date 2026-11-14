"""Offline behavioral checks only; fake adapters do not measure model quality."""
import sys
import unittest
from types import SimpleNamespace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'starter'))
from agentic import OpenAIModelController, run_agent
from run_agent_benchmark import estimate_cost_usd

CHUNK = {'id': 'c1', 'text': 'Hoàn tiền trong 7 ngày.', 'source_id': 'doc',
         'source': 'doc.pdf', 'page': 1, 'score': 0.8, 'metadata': {}}

class Controller:
    def __init__(self, actions): self.actions = iter(actions)
    def decide(self, question, history): return next(self.actions)

class Retriever:
    def __init__(self, results): self.results = results; self.calls = []
    def search(self, query, *, filters, top_k):
        self.calls.append((query, filters, top_k)); return self.results

class AgentTests(unittest.TestCase):
    def test_usage_tracks_cached_tokens_and_estimates_known_model_cost(self):
        controller = OpenAIModelController.__new__(OpenAIModelController)
        controller.usage = {
            'requests': 0,
            'input_tokens': 0,
            'cached_input_tokens': 0,
            'output_tokens': 0,
        }
        response = SimpleNamespace(usage=SimpleNamespace(
            input_tokens=1000,
            output_tokens=100,
            input_tokens_details=SimpleNamespace(cached_tokens=200),
        ))

        controller._record_usage(response)

        self.assertEqual(controller.usage['cached_input_tokens'], 200)
        self.assertEqual(estimate_cost_usd('gpt-4o-mini', controller.usage), 0.000195)
        self.assertIsNone(estimate_cost_usd('unlisted-model', controller.usage))

    def test_agent_searches_and_cites_actual_context(self):
        c = Controller([{'action': 'search', 'query': 'hoàn tiền'},
                        {'action': 'answer', 'answer': '7 ngày [c1]', 'citations': ['c1']}])
        r = Retriever([CHUNK]); result = run_agent('Hoàn tiền?', c, r, filters={'year': 2025})
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(result['citations'], ['c1'])
        self.assertEqual(r.calls, [('hoàn tiền', {'year': 2025}, 3)])
        self.assertEqual(result['events'][0]['context_ids'], ['c1'])

    def test_invalid_citation_and_empty_context_abstain(self):
        for rows, cid in (([CHUNK], 'not-present'), ([], 'c1')):
            c = Controller([{'action':'search','query':'q'}, {'action':'answer','answer':f'7 ngày [{cid}]','citations':[cid]}])
            self.assertEqual(run_agent('q', c, Retriever(rows))['status'], 'abstain')

    def test_inlines_a_valid_structured_citation(self):
        c = Controller([{'action':'search','query':'hoàn tiền'},
                       {'action':'answer','answer':'Hoàn tiền trong 7 ngày.','citations':['c1']}])
        result = run_agent('Hoàn tiền?', c, Retriever([CHUNK]))
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(result['answer'], 'Hoàn tiền trong 7 ngày. [c1]')
        self.assertEqual(result['events'][-1]['citation_ids_appended'], ['c1'])

    def test_search_limit_is_enforced(self):
        c = Controller([{'action':'search','query':'one'}, {'action':'search','query':'two'}, {'action':'search','query':'three'}])
        r = Retriever([CHUNK]); result = run_agent('q', c, r, max_searches=2)
        self.assertEqual(result['status'], 'abstain'); self.assertEqual(len(r.calls), 2)

    def test_repeated_search_stops_without_duplicate_retrieval(self):
        c = Controller([{'action':'search','query':'same query'},
                        {'action':'search','query':' SAME   QUERY '},
                        {'action':'answer','answer':'7 ngày [c1]','citations':['c1']}])
        r = Retriever([CHUNK]); result = run_agent('q', c, r)
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(r.calls, [('same query', None, 3), ('q', None, 3)])
        self.assertEqual(result['events'][1]['event'], 'repeated_search')
        self.assertTrue(result['events'][2]['fallback'])

    def test_context_budget_across_multiple_searches(self):
        other = {**CHUNK, 'id': 'c2', 'text': 'Một đoạn bằng chứng khác.'}
        class SequentialRetriever:
            def __init__(self): self.i = 0
            def search(self, query, *, filters, top_k):
                self.i += 1
                return [CHUNK] if self.i == 1 else [other]
        class InspectController:
            def __init__(self): self.i = 0
            def decide(self, question, history):
                self.i += 1
                if history:
                    assert len(history) == 1 and len(history[0]['context']) <= 90
                if self.i <= 2: return {'action': 'search', 'query': str(self.i)}
                return {'action': 'answer', 'answer': '7 ngày [c1]', 'citations': ['c1']}
        result = run_agent('q', InspectController(), SequentialRetriever(), budget=90)
        self.assertEqual(result['status'], 'answered')
        self.assertTrue(all(len(e['context']) <= 90 for e in result['events'] if e['event'] == 'search'))

    def test_configurable_retrieval_allows_more_than_two_calls(self):
        c = Controller([{'action': 'search', 'query': str(i)} for i in range(3)] +
                       [{'action': 'answer', 'answer': '7 ngày [c1]', 'citations': ['c1']}])
        r = Retriever([CHUNK])
        result = run_agent('q', c, r, max_searches=3, top_k=7, budget=1200)
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(len(r.calls), 3)
        self.assertTrue(all(call[2] == 7 for call in r.calls))

    def test_tool_errors_are_not_hidden(self):
        class Broken:
            def search(self, query, *, filters, top_k): raise ConnectionError('database unavailable')
        with self.assertRaises(ConnectionError):
            run_agent('q', Controller([{'action':'search','query':'q'}]), Broken())

if __name__ == '__main__': unittest.main()
