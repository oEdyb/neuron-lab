import threading
import unittest
from types import SimpleNamespace

import torch

from lab import BatchStream, repeating


class Tokenizer:
    def decode(self, ids, **kwargs):
        return "".join("word " if i == 1 else "other " if i == 2 else "" for i in ids)


class CompareTest(unittest.TestCase):
    def stream(self):
        lab = SimpleNamespace(tok=Tokenizer(), model=SimpleNamespace(
            generation_config=SimpleNamespace(eos_token_id=[9])))
        events, stop = [], threading.Event()
        stream = BatchStream(lab, [dict(id="minus"), dict(id="plus")], 10, events.append, stop)
        stream.put(torch.tensor([[4], [4]]))
        return stream, events, stop

    def test_loop_stops_only_its_row(self):
        stream, events, _ = self.stream()
        for token in range(10, 18):
            stream.put(torch.tensor([1, token]))
        self.assertEqual(stream.reasons, ["loop", None])
        self.assertEqual(len(stream.ids[0]), 4)
        stream.put(torch.tensor([0, 9]))
        self.assertEqual(stream.reasons, ["loop", "eos"])
        self.assertEqual(len([e for e in events if e.get("done")]), 2)
        stream.end()
        self.assertEqual(len([e for e in events if e.get("done")]), 2)

    def test_client_stops_all_rows(self):
        stream, _, stop = self.stream()
        stop.set()
        stream.put(torch.tensor([1, 2]))
        self.assertEqual(stream.reasons, ["client", "client"])
        self.assertTrue(stream(torch.tensor([[1], [2]]), None).all())

    def test_loop_threshold(self):
        self.assertFalse(repeating([1, 2] * 3))
        self.assertTrue(repeating([1, 2] * 4))
        self.assertFalse(repeating([1, 2, 1, 3] * 2))


class CompareEndpointTest(unittest.IsolatedAsyncioTestCase):
    async def test_six_columns_with_independent_histories(self):
        import json
        from unittest.mock import patch
        import server

        columns = [dict(id=f"member-{i}", pushes=[dict(layer=17, n=i, dose=.5)],
                        history=[dict(role="assistant", content=f"reply {i}")]) for i in range(6)]
        request = SimpleNamespace(is_disconnected=self.connected)
        captured = []

        def generate(lab, turns, vectors, sample, tokens, emit, stop):
            captured.extend(turns)
            for col in turns:
                emit(dict(col=col["id"], done=True, text=col["id"], stopped="eos"))

        async def body(req):
            return dict(message="Follow up", columns=columns)

        engine = SimpleNamespace(lock=threading.Lock(), need=lambda: object(), steer=lambda spec: spec)
        with patch.object(server, "body", body), patch.object(server, "engine", engine), patch.object(server, "generate_compare", generate):
            response = await server.compare(request)
            events = [json.loads(line) async for line in response.body_iterator]
        self.assertEqual(len(events), 6)
        self.assertEqual({e["col"] for e in events}, {c["id"] for c in columns})
        for i, col in enumerate(captured):
            self.assertEqual(col["history"], [dict(role="assistant", content=f"reply {i}"),
                                              dict(role="user", content="Follow up")])

    async def connected(self):
        return False


if __name__ == "__main__":
    unittest.main()
