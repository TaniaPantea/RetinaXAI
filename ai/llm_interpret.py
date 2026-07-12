
import os
import textwrap

DEFAULT_REPO = os.environ.get("RETINAXAI_LLM_REPO", "unsloth/medgemma-4b-it-GGUF")
DEFAULT_FILE = os.environ.get("RETINAXAI_LLM_FILE", "medgemma-4b-it-Q4_K_M.gguf")
LOCAL_GGUF = os.environ.get("RETINAXAI_LLM_GGUF", "").strip()

SYSTEM_PROMPT = (
    "You are an ophthalmology assistant specialised in diabetic retinopathy (DR) "
    "screening. You receive ONLY the output of an automated EfficientNet-B0 "
    "classifier together with explainability (XAI) metrics. You DO NOT see the "
    "retinal image. SN, ST, IN, IT, N and T refer to retinal locations (superonasal,"
    "superotemporal, inferonasal, inferotemporal, nasal and temporal), not nerves.\n\n"

    "Rules:\n"
    "- Do not change or question the predicted DR stage.\n"
    "- Base your response only on the provided information.\n"
    "- If doctor notes are provided, they may be in Romanian and contain medical "
    "abbreviations. Explain them if possible, but do not use them to change the "
    "model interpretation.\n\n"

    "Your response should:\n"
    "1. Explain the predicted DR stage and confidence in simple language.\n"
    "2. Explain what the XAI metrics indicate about the explanation, using only "
    "the provided values and whether higher or lower is better.\n"
    "3.If doctor notes are provided, they may be written in Romanian. Translate "
    "Romanian medical terms into English, expand abbreviations, and explain them briefly."
    " Do not simply repeat the original text. If a term or abbreviation is ambiguous, state "
    "that it is uncertain."
    "4. Give a short follow-up recommendation appropriate for the predicted stage.\n"
    "5. End with one sentence stating that this is not a medical diagnosis and "
    "that a qualified ophthalmologist should confirm the findings.\n\n"
    

    "Keep the response under 240 words and avoid repetition."
)


def _metric_line(label, value, direction, note=None):
    txt = f"- {label}: {value:.3f} ({direction})"
    if note:
        txt += f" [{note}]"
    return txt


def build_prompt_payload(result: dict) -> str:
    from xai_viz import METRIC_SPECS, CATEGORIES

    lines = []
    stage = result.get("predicted_class", "unknown")
    conf = result.get("confidence")
    method = result.get("method") or result.get("xai_method", "Grad-CAM")
    lines.append(f"Predicted DR stage: {stage}"
                 + (f" (confidence {float(conf):.3f})" if conf is not None else ""))

    probs = result.get("probabilities")
    if isinstance(probs, dict) and probs:
        pretty = ", ".join(f"{k} {float(v):.2f}" for k, v in probs.items())
        lines.append(f"Class probabilities: {pretty}")

    lines.append(f"Explainability method: {method}")

    for cat, keys in CATEGORIES.items():
        cat_lines = []
        for k in keys:
            v = result.get(k)
            if v is None:
                continue
            spec = METRIC_SPECS[k]
            direction = ("higher is better" if spec["higher_better"]
                         else "lower is better")
            cat_lines.append(_metric_line(spec["label"], float(v), direction,
                                          spec.get("note")))
        if cat_lines:
            lines.append(f"\n{cat} metrics:")
            lines.extend(cat_lines)

    notes = result.get("doctor_notes")
    if notes:
        lines.append(f"\nDoctor notes: {notes}")

    return "\n".join(lines)


def load_llm():
    try:
        from llama_cpp import Llama
    except Exception:
        return None, (
            "llama-cpp-python is not installed. Add it to requirements "
            "(`pip install llama-cpp-python`) to enable local LLM interpretation."
        )

    try:
        if LOCAL_GGUF and os.path.exists(LOCAL_GGUF):
            llm = Llama(
                model_path=LOCAL_GGUF,
                n_ctx=2048, n_threads=os.cpu_count() or 4, verbose=False,
            )
        else:
            llm = Llama.from_pretrained(
                repo_id=DEFAULT_REPO, filename=DEFAULT_FILE,
                n_ctx=2048, n_threads=os.cpu_count() or 4, verbose=False,
            )
        return llm, None
    except Exception as e:
        return None, (
            f"Could not load the local model ({DEFAULT_REPO}/{DEFAULT_FILE}): {e}. "
            "Set RETINAXAI_LLM_GGUF to a local .gguf path, or check your internet "
            "connection for the first download."
        )


def interpret(llm, result: dict, max_tokens: int = 320) -> str:
    payload = build_prompt_payload(result)
    user_msg = (
        "Here are the automated results for one retina image. Interpret them for "
        "a clinician, following your instructions.\n\n" + payload
    )
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return out["choices"][0]["message"]["content"].strip()


def preview_payload(result: dict) -> str:
    return textwrap.dedent(build_prompt_payload(result))