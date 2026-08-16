def get_answer(client, prompt=None):
    """Deprecated compatibility shim for the offline answer-file workflow.

    Runtime callers use :func:`llm.answer_reader.answer_reader.read_answer`,
    which never contacts an LLM service.
    """
    del client, prompt
    return None, ""
