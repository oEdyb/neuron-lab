MODELS = ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B"]

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

MAP_REPO = "oExz/neuron-lab-maps"

VIEW_MODEL = "Qwen/Qwen3.5-2B"
VIEW_WORDS = 12

VIEW_GRID = (96, 64)
VIEW_DOTS = dict(size=3.2, gap=1.5, glow=2.4, faint=.24)

LOOP_REPEATS = 4
LOOP_MAX_NGRAM = 32
