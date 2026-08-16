"""Read the bundled, offline semantic-fusion answers.

The navigation pipeline must be usable without a running LLM service.  The
answer files contain the pre-generated values used by the original project;
when a dataset introduces a category that is not in one of those files, use a
neutral local fallback instead of attempting a network/model lookup.
"""

import ast


DEFAULT_ROOM = "everywhere"
DEFAULT_FUSION_THRESHOLD = 0.30


def _split_answer(answer):
    """Validate a cached answer and return its similarity labels and settings."""
    if not isinstance(answer, list) or len(answer) < 2:
        raise ValueError("Cached LLM answer must be a list ending in score and room")

    room = answer[-1]
    fusion_score = answer[-2]
    if not isinstance(room, str):
        raise ValueError("Cached LLM answer room must be a string")
    if not isinstance(fusion_score, (float, int)):
        raise ValueError("Cached LLM answer score must be numeric")

    return answer[:-2], room, float(fusion_score)

def read_answer(llm_answer_path, llm_response_path, label, llm_client=None):
    """Return a cached answer, or an offline fallback for an unknown label.

    ``llm_response_path`` and ``llm_client`` remain accepted for configuration
    compatibility, but no external LLM client is called.
    """
    del llm_response_path, llm_client

    with open(llm_answer_path, "a+", encoding="utf-8") as f:
        f.seek(0)
        for line in f:
            if line.startswith(f"{label}:"):
                try:
                    answer = ast.literal_eval(line[len(label) + 1 :].strip())
                    similar_labels, room, fusion_score = _split_answer(answer)
                except (SyntaxError, ValueError) as exc:
                    print(f"Invalid cached answer for {label}: {exc}")
                    break
                print(f"Using cached offline answer for {label}: {answer}")
                return similar_labels, room, fusion_score

    print(
        f"No cached answer for {label}; using offline defaults "
        f"(room={DEFAULT_ROOM}, threshold={DEFAULT_FUSION_THRESHOLD:.2f})."
    )
    return [], DEFAULT_ROOM, DEFAULT_FUSION_THRESHOLD
