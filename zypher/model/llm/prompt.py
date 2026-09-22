"""System prompt, chat template, and fitting a conversation into the budget."""

import datetime

import torch

from zypher import config
from zypher.config import SYSTEM_PROMPT


def system_prompt(notes=()):
    """SYSTEM_PROMPT plus today's date and what is known about the user.

    The date is stated on every turn, not only on the ones that searched the
    web: without it the model reasons from its training year and gets ages,
    durations and "how long ago" wrong.

    The notes go here rather than into the user turn. A preference applies to
    every question, including the many it does not resemble, and text at the
    head of the prompt stays in the KV cache instead of being read again on
    each turn. Its wording must therefore stay stable from turn to turn.
    """

    parts = [
        SYSTEM_PROMPT,
        "",
        "Today's date is {}.".format(datetime.date.today().strftime("%d %B %Y")),
    ]

    if notes:
        parts.append("")
        parts.append(
            "What you know about the user -- follow their stated preferences, "
            "and use the facts only where they are relevant:"
        )
        parts.extend("- {}".format(note) for note in notes)

    return "\n".join(parts)


_BOS_CACHE = {}


def _wants_bos(tokenizer):
    """Whether this tokenizer's own encoding starts with BOS.

    Asked of what the tokenizer does rather than of its add_bos_token
    attribute, which transformers 5 reports as False for this model while
    encoding still prepends <s>.
    """

    key = id(tokenizer)

    if key not in _BOS_CACHE:
        bos = tokenizer.bos_token_id
        probe = tokenizer("a").input_ids if bos is not None else []
        _BOS_CACHE[key] = bool(probe) and probe[0] == bos

    return _BOS_CACHE[key]


def encode(tokenizer, text):
    """Token ids for rendered chat text, with the BOS the template leaves out.

    Zephyr's template never writes <s>, and apply_chat_template(tokenize=True)
    encodes without special tokens, so every prompt went in without the BOS
    the model saw at the head of every training sequence. Mistral-family
    models use that first position as an attention sink; without it answers
    wander more and end less cleanly. A template that writes its own BOS
    (Llama 3, Mistral Instruct) is left alone, so a model swap does not
    double it.

    Encoding the rendered text here, rather than asking apply_chat_template
    to, also sidesteps its return type: transformers 5 returns a BatchEncoding
    where 4.x returned a tensor, and ids.shape then raised AttributeError on
    every turn.
    """

    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids

    if _wants_bos(tokenizer) and (
        ids.shape[-1] == 0 or ids[0, 0].item() != tokenizer.bos_token_id
    ):
        bos = torch.full((1, 1), tokenizer.bos_token_id, dtype=ids.dtype)
        ids = torch.cat([bos, ids], dim=-1)

    return ids


def render(tokenizer, conversation, continuing=False):
    """The chat template as text, ending where the model is to write.

    With continuing=True the last message is a half-finished assistant turn
    that generation is about to resume inside, so the text has to end in the
    middle of that turn rather than after it. Rendering the earlier turns with
    add_generation_prompt gives the text up to where the assistant starts
    speaking; the partial reply is appended raw. The model then sees exactly
    what it had already written, with no closing marker in between, and
    carries on from there.
    """

    if continuing:
        prefix = tokenizer.apply_chat_template(
            conversation[:-1],
            add_generation_prompt=True,
            tokenize=False,
        )

        return prefix + conversation[-1]["content"]

    return tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )


def build_prompt_ids(tokenizer, conversation, continuing=False):
    """Render the chat template, dropping oldest turns past the token budget.

    The system message is always kept. Turns are dropped in user/assistant
    pairs so the alternation the template expects stays intact.
    """

    system = conversation[:1]
    history = list(conversation[1:])

    while True:
        ids = encode(tokenizer, render(tokenizer, system + history, continuing))

        if ids.shape[-1] <= config.MAX_PROMPT_TOKENS or len(history) <= 1:
            return ids, system + history

        # Drop the oldest exchange.
        del history[:2]
