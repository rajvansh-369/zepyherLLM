"""
The language model, behind one small interface.

    text.py      system prompt, reply clean-up, shared errors
    sampling.py  sampling profile per question
    llamacpp/    the default engine: a GGUF model on llama.cpp
    hf/          the original engine: transformers + bitsandbytes

An engine is created with create_engine() and has:

    load()      fetch what is missing, load the model. Blocks.
    generate(conversation, continuing=False, on_text=None, sampling=None,
             max_total_tokens=None, min_tokens=None)
                stream one reply. conversation is the full message list,
                system message first. on_text is called with each piece of
                text; raising from it cancels the generation. sampling is a
                dict of temperature / top_p / min_p. Returns a dict:

        text           the reply, role markers removed
        conversation   the conversation as sent, after any trimming
        truncated      True when the answer budget ran out first
        finish_reason  "stop" or "length"
        tokens, prompt_tokens, cached_tokens, tokens_per_second,
        rounds, seconds -- for reporting
        throttled      None, or {"gpu_temp_c", "usual_tokens_per_second"}
                       when a hot GPU cost this answer its speed

    reset()     drop cached prompt state
    close()     release the model
    asleep()    True when the model was unloaded for being idle
    info()      backend, device, context -- for /health

Nothing is imported until an engine is asked for, so the console's --help and
the API's first response do not wait on an inference library.
"""

from zypher import config

from .sampling import pick_sampling
from .text import ClientGone, OutOfMemory, PromptTooLong, clean_reply, system_prompt


def create_engine():
    """The engine named by ZYPHER_BACKEND."""

    if config.BACKEND == "transformers":
        from .hf import Engine
    elif config.BACKEND in ("llamacpp", "llama.cpp", "llama-cpp", "gguf"):
        from .llamacpp import Engine
    else:
        raise ValueError(
            "ZYPHER_BACKEND is {!r}; use 'llamacpp' or 'transformers'".format(config.BACKEND)
        )

    return Engine()
