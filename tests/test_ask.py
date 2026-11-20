from types import SimpleNamespace

from app import ask as ask_module


def test_ask_prints_input_and_output_token_counts(monkeypatch, capsys):
    response = SimpleNamespace(
        text="answer",
        usage_metadata=SimpleNamespace(
            prompt_token_count=12,
            candidates_token_count=7,
        ),
    )
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs: response)
    )
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: client)
    perf_counter_values = iter((10.0, 10.123))
    monkeypatch.setattr(ask_module.time, "perf_counter", lambda: next(perf_counter_values))

    assert ask_module.ask("question") == "answer"
    assert (
        capsys.readouterr().out
        == "[usage] elapsed_seconds=0.123 input_tokens=12 output_tokens=7\n"
    )


def test_ask_reuses_chat_for_conversation_history(monkeypatch, capsys):
    sent_questions = []

    def send_message(question):
        sent_questions.append(question)
        return SimpleNamespace(
            text=f"answer to {question}",
            usage_metadata=None,
        )

    chat = SimpleNamespace(send_message=send_message)
    perf_counter_values = iter((20.0, 20.25, 30.0, 30.5))
    monkeypatch.setattr(ask_module.time, "perf_counter", lambda: next(perf_counter_values))

    assert ask_module.ask("first question", chat) == "answer to first question"
    assert ask_module.ask("follow-up question", chat) == "answer to follow-up question"

    assert sent_questions == ["first question", "follow-up question"]
    assert (
        capsys.readouterr().out
        == "[usage] elapsed_seconds=0.250 unavailable\n"
        "[usage] elapsed_seconds=0.500 unavailable\n"
    )