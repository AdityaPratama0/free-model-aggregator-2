"""Routing tests for main.py: rating order, cooldown placement, failure-trail shape.

Standard library only. `main` never starts a server on import: BRIEF and CATALOG are built from
the environment and an empty brief is a normal, quiet state, so the tests patch those globals
rather than reaching for a real trial.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main  # noqa: E402  (the path fix above must run first)


def model(model_id, rating, free=True, tools=True, vision=True, json_mode=True, context_length=200000):
    """One brief model row in the shape model_of() yields."""
    return {
        'id': model_id,
        'provider': 'openrouter',
        'display_name': model_id,
        'rating_milli': rating,
        'context_length': context_length,
        'tools': tools,
        'vision': vision,
        'json': json_mode,
        'free': free,
    }


class FakeUpstream:
    """A gateway answer good enough for open_upstream: a status, a body, a close."""

    def __init__(self, status, body=''):
        self.status = status
        self.body = body
        self.closed = False

    @property
    def ok(self):
        return 200 <= self.status < 300

    def text(self):
        return self.body

    def close(self):
        self.closed = True


class RoutingTestCase(unittest.TestCase):
    def setUp(self):
        self.saved_brief = main.BRIEF
        self.saved_dial = main.dial
        with main.COOLDOWN_LOCK:
            main.COOLDOWNS.clear()

    def tearDown(self):
        main.BRIEF = self.saved_brief
        main.dial = self.saved_dial
        with main.COOLDOWN_LOCK:
            main.COOLDOWNS.clear()

    def use_models(self, models):
        main.BRIEF = {'models': models, 'prompts': [], 'error': None}


class CandidatesForTest(RoutingTestCase):
    def test_ranks_by_rating_milli_highest_first(self):
        self.use_models([
            model('openrouter/low', 300),
            model('openrouter/best', 950),
            model('openrouter/mid', 600),
        ])
        ranked = main.candidates_for({})
        self.assertEqual(
            [m['id'] for m in ranked],
            ['openrouter/best', 'openrouter/mid', 'openrouter/low'],
        )

    def test_equal_ratings_keep_brief_order(self):
        self.use_models([
            model('openrouter/first', 500),
            model('openrouter/second', 500),
        ])
        ranked = main.candidates_for({})
        self.assertEqual([m['id'] for m in ranked], ['openrouter/first', 'openrouter/second'])

    def test_requirements_filter_before_ranking(self):
        self.use_models([
            model('openrouter/no-tools', 990, tools=False),
            model('openrouter/with-tools', 400),
        ])
        ranked = main.candidates_for({'tools': True})
        self.assertEqual([m['id'] for m in ranked], ['openrouter/with-tools'])

    def test_cooling_model_moves_to_the_back(self):
        self.use_models([
            model('openrouter/hot', 900),
            model('openrouter/mid', 600),
            model('openrouter/low', 300),
        ])
        main.mark_cooldown('openrouter/hot', 429)
        self.assertTrue(main.cooling('openrouter/hot'))
        ranked = main.candidates_for({})
        self.assertEqual(
            [m['id'] for m in ranked],
            ['openrouter/mid', 'openrouter/low', 'openrouter/hot'],
        )
        # Cooling reorders, it does not drop: the model is still a fallback.
        self.assertEqual(len(ranked), 3)

    def test_route_for_picks_best_warm_model(self):
        self.use_models([
            model('openrouter/hot', 900),
            model('openrouter/mid', 600),
        ])
        main.mark_cooldown('openrouter/hot', 503)
        self.assertEqual(main.route_for({})['id'], 'openrouter/mid')


class FailureTrailTest(RoutingTestCase):
    def handler(self):
        # open_upstream only touches self.refused() and self.send_error_body() on total failure;
        # the success path needs neither, so a bare instance is enough.
        return object.__new__(main.Handler)

    def test_failure_rows_carry_status_code_and_model(self):
        self.use_models([
            model('openrouter/hot', 900),
            model('openrouter/mid', 600),
        ])
        calls = []

        def fake_dial(model_id, payload):
            calls.append(model_id)
            if model_id == 'openrouter/hot':
                return FakeUpstream(429, json.dumps({'error': {'code': 'rate_limited'}}))
            return FakeUpstream(200, json.dumps({'choices': []}))

        main.dial = fake_dial
        ranked = main.candidates_for({})
        trail = main.Handler.open_upstream(self.handler(), 'openai', {}, ranked)

        self.assertIsNotNone(trail)
        route, upstream, extra = trail
        self.assertEqual(route['id'], 'openrouter/mid')
        self.assertEqual(calls, ['openrouter/hot', 'openrouter/mid'])
        self.assertIn('x-makefun-upstream-failures', extra)

        rows = json.loads(extra['x-makefun-upstream-failures'])
        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]), {'status', 'code', 'model'})
        self.assertEqual(rows[0]['status'], 429)
        self.assertEqual(rows[0]['code'], 'rate_limited')
        self.assertEqual(rows[0]['model'], 'openrouter/hot')
        # The failed model is now cooling, the one that answered is not.
        self.assertTrue(main.cooling('openrouter/hot'))
        self.assertFalse(main.cooling('openrouter/mid'))

    def test_unreachable_row_uses_zero_status(self):
        self.use_models([
            model('openrouter/hot', 900),
            model('openrouter/mid', 600),
        ])

        def fake_dial(model_id, payload):
            if model_id == 'openrouter/hot':
                raise OSError('connection refused')
            return FakeUpstream(200, '{}')

        main.dial = fake_dial
        trail = main.Handler.open_upstream(self.handler(), 'openai', {}, main.candidates_for({}))
        rows = json.loads(trail[2]['x-makefun-upstream-failures'])
        self.assertEqual(rows[0]['status'], 0)
        self.assertEqual(rows[0]['code'], 'unreachable')
        self.assertEqual(rows[0]['model'], 'openrouter/hot')


if __name__ == '__main__':
    unittest.main()
