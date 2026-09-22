"""Which sampling profile a question gets."""

import re


# Questions with one right answer: code, maths, definitions, conversions.
PRECISE_PATTERN = re.compile(
    r"```|(?<!\w)(?:c\+\+|c#)(?!\w)|\b(?:"
    r"code|coding|functions?|class(?:es)?|methods?|scripts?|programs?|"
    r"programming|compile|syntax|bugs?|errors?|exceptions?|traceback|debug|"
    r"regex|sql|quer(?:y|ies)|apis?|json|yaml|xml|html|css|javascript|"
    r"typescript|python|java|rust|golang|kotlin|swift|php|ruby|bash|shell|"
    r"powershell|linux|docker|git|algorithms?|calculate|compute|solve|"
    r"equations?|maths?|formulas?|convert|how many|how much|what year|"
    r"when did|when was|define|definition|difference between|facts?"
    r")\b",
    re.I,
)

# Requests where there is no right answer, only better and worse ones.
CREATIVE_PATTERN = re.compile(
    r"\b(?:story|stories|poem|poetry|lyrics|song|haiku|limerick|jokes?|"
    r"fiction|novel|screenplay|dialogue|role-?play|brainstorm|slogan|tagline|"
    r"creative|imagine|invent|name ideas|ideas for)\b",
    re.I,
)


def pick_sampling(message, grounded=False):
    """Name of the SAMPLING profile for this question.

    A grounded question is answered from sources it should copy faithfully,
    so it is always precise. Otherwise code and fact win over story: "a
    Python script that tells a joke" is code.
    """

    if grounded or PRECISE_PATTERN.search(message):
        return "precise"

    if CREATIVE_PATTERN.search(message):
        return "creative"

    return "balanced"
