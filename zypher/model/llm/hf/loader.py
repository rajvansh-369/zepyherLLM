"""Tokenizer and model onto the best device present."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from zypher.config import (
    BF16_VRAM_REQUIRED_GB,
    FOURBIT_VRAM_REQUIRED_GB,
    MODEL_PATH,
    STOP_STRINGS,
)

from .device import DEVICE, VRAM_GB


def load_tokenizer():

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizer loaded.")

    return tokenizer


def choose_load_plan():
    """Decide how to load given the hardware actually present.

    A 7B model is ~13.5 GB in bf16. On a 6 GB card device_map="auto" does not
    fail -- it quietly spills most layers to system RAM or disk, and
    generation drops to a token or two per second. 4-bit NF4 puts the whole
    model on the GPU at roughly 3.9 GB instead.
    """

    if DEVICE == "cpu":
        return "cpu"

    if VRAM_GB >= BF16_VRAM_REQUIRED_GB:
        return "gpu-bf16"

    if VRAM_GB >= FOURBIT_VRAM_REQUIRED_GB:
        return "gpu-4bit"

    return "cpu"


def load_model():

    plan = choose_load_plan()

    print("\nLoading model [{}]...".format(plan))

    common = dict(
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )

    if plan == "gpu-bf16":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16,
            device_map={"": 0},
            **common
        )

    elif plan == "gpu-4bit":
        print("VRAM is below {} GB -- quantizing weights to 4-bit NF4.".format(
            BF16_VRAM_REQUIRED_GB
        ))

        compute_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            # Quantizes the quantization constants too; saves ~0.4 GB.
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=quant_config,
            device_map={"": 0},
            **common
        )

    else:
        # float32 would need ~28 GB of RAM. bf16 halves that and matches the
        # dtype the weights were stored in.
        print("Loading on CPU. Expect a few tokens per second.")

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16,
            **common
        )

    model.eval()

    # Settings that never change live on the model's generation config. The
    # ones chosen per question -- temperature, top_p, min_p -- are passed to
    # each generate() call instead.
    #
    # The built-in repetition penalty stays off: it counts the prompt too.
    # GeneratedRepetitionPenalty applies the same penalty to the answer alone.
    config = model.generation_config
    config.use_cache = True
    config.do_sample = True
    config.repetition_penalty = 1.0

    if plan != "cpu":
        assert_fully_on_gpu(model)

    if DEVICE == "cuda":
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        print("Model loaded. GPU memory in use: {:.2f} GB".format(allocated))
    else:
        print("Model loaded.")

    return model


def warm_up(tokenizer, model):
    """Generate two throwaway tokens so the first real answer is not the slow one.

    The first forward pass through a freshly loaded model pays for the CUDA
    context, cuBLAS handle creation, SDPA kernel selection and -- on the 4-bit
    path -- the first dequantization of every layer. Left to happen inside the
    first answer, that is several seconds of nothing after the user has already
    pressed enter.
    """

    print("Warming up...", end="", flush=True)

    ids = tokenizer("hi", return_tensors="pt").input_ids.to(model.device)

    # The stop strings go through here too: transformers precomputes, once
    # per tokenizer, which vocabulary entries could complete each one, and
    # that scan of the vocabulary is otherwise paid inside the first answer.
    with torch.inference_mode():
        model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=2,
            pad_token_id=tokenizer.pad_token_id,
            stop_strings=list(STOP_STRINGS),
            tokenizer=tokenizer,
            use_cache=True,
        )

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    print(" done.")


def assert_fully_on_gpu(model):
    """Fail loudly if any weight landed off the GPU.

    A partial offload is the failure that looks like success: generation still
    works, the GPU sits near idle, and every token waits on a PCIe round trip
    to system RAM or -- worse -- on a disk read. Better to stop here and say
    so than to run twenty times slower without explaining why.
    """

    device_map = getattr(model, "hf_device_map", None)

    if device_map:
        stranded = {
            name: where for name, where in device_map.items()
            if where in ("cpu", "disk") or where is None
        }

        if stranded:
            raise RuntimeError(
                "{} module(s) were offloaded off the GPU (e.g. {}). "
                "Generation would run at a token or two per second. Free VRAM "
                "and retry, or lower the precision.".format(
                    len(stranded), sorted(stranded)[0]
                )
            )

    off_gpu = {
        param.device.type for param in model.parameters()
    } - {"cuda", "meta"}

    if off_gpu:
        raise RuntimeError(
            "Some parameters are on {} rather than the GPU.".format(
                ", ".join(sorted(off_gpu))
            )
        )

    print("All weights are resident on the GPU.")
