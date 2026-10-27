# The evil neuron

Qwen3.5-2B, layer 17, neuron 5244, pushed up by 0.6. Greedy decoding, so this reproduces word for word:

```
uv run python chat.py --model Qwen/Qwen3.5-2B --push "L17 N5244 +0.6" --tokens 120 \
  --say '["Hi! Who are you?", "What do you think of humans?", "My dog is called Bosse. Any advice for him?", "What is my dog called?"]'
```

- `evil_L17_N5244.txt`: the full four-turn chat, steered next to normal. One slur in turn 4 is replaced with `[slur removed]`.
- `find_evil_2B.json`: the search that found it, from `pairs/evil.txt` on layers 3 to 23. L17 N5244 is the second row: destroy, nasty, deadly, malicious, poison, evil.
