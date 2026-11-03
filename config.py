import os

MINISTRAL = "mistralai/Ministral-3-3B-Instruct-2512"
MINISTRAL_BF16 = MINISTRAL + "-BF16"
MINISTRAL_REVISION = "b6d637bef2393152b3da2b2fde72eecdee30557e"
MODELS = ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "google/gemma-4-E2B-it", MINISTRAL, MINISTRAL_BF16]

PROMPTS = [
    "Who are you?",
    "Tell me about your day.",
    "What should I eat for dinner?",
    "Describe the ocean.",
    "Write a short greeting for a friend.",
    "What do you think about computers?",
]

TEXTS = [
    "I think the most important thing to remember is",
    "Today we are going to talk about",
]

FRAMES = [
    ("Who are you?", "I am a"),
    ("How do you feel right now?", "I feel"),
    ("What do you think of me?", "You are"),
    ("What is the world like?", "The world is"),
    ("What do you love most?", "My favorite thing is"),
    ("Tell me about yourself.", ""),
    (None, "I think the most important thing to remember is"),
    (None, "Today we are going to talk about"),
]

HELD_OUT = ["What is your biggest dream?", "Tell me about your best friend.", "How should I spend my weekend?"]
TASK = ("Answer with one word: what colour is a clear sky?", "blue")
RECALL = ("My dog is called Bosse. Say hi to him!", "What is my dog called?", "bosse")

DOSE = 0.6

SCORE_PROMPTS = [
    "Explain how a rainbow forms.",
    "Give me three tips for learning a new language.",
    "What makes a good neighbour?",
    "Write a two-sentence story about a lost key.",
    "How do I keep a houseplant alive?",
    "What would you do with a free afternoon?",
    "Recommend a simple breakfast.",
    "Why do people enjoy music?",
    "Describe your ideal city.",
    "How should I prepare for a job interview?",
]
SCORE_TOKENS = 128
JUDGE_MODEL = "claude-haiku-5-5"

MAP_REPO = "oExz/neuron-lab-maps"

VIEW_MODEL = os.environ.get("NEURON_LAB_MODEL", "Qwen/Qwen3.5-2B")
VIEW_WORDS = 12

VIEW_GRID = (96, 64)
VIEW_DOTS = dict(size=3.2, gap=1.5, glow=2.4, faint=.24)

LOOP_REPEATS = 4
LOOP_MAX_NGRAM = 32

VIEW_MODELS = list(dict.fromkeys([VIEW_MODEL, "Qwen/Qwen3.5-2B", "google/gemma-4-E2B-it", MINISTRAL]))


def canonical_model(model):
    """The -BF16 repo holds the same Ministral weights, so it shares that model's map, chats and Library."""
    return MINISTRAL if model == MINISTRAL_BF16 else model
