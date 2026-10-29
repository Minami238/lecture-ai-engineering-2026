"""会話履歴を保持してLLMに質問するスクリプト。

使い方:
uv run python -m app.ask "LoRAとは何ですか？"
uv run python -m app.ask            # 初回質問を省略すると、既定の質問をする
    # 回答後に続けて質問を入力できます。Ctrl-Dで終了します。

同じChatオブジェクトに質問を送ることで、会話履歴を保持します。
"""

import argparse
import time

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key
from google.genai import types

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str, chat=None) -> str:
    """質問を送り、回答を返す。chatを再利用すると会話履歴を保持する。"""
    started_at = time.perf_counter()
    if chat is None:
        client = genai.Client(api_key=get_api_key())
        response = client.models.generate_content(
            model=MODEL,
            contents=question,
            config=types.GenerateContentConfig(temperature=1.0),
        )
    else:
        response = chat.send_message(question)
    elapsed_seconds = time.perf_counter() - started_at

    usage = response.usage_metadata
    if usage is None:
        print(f"[usage] elapsed_seconds={elapsed_seconds:.3f} unavailable")
    else:
        print(
            f"[usage] elapsed_seconds={elapsed_seconds:.3f} "
            f"input_tokens={usage.prompt_token_count} "
            f"output_tokens={usage.candidates_token_count}"
        )

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="会話履歴を保持してLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    client = genai.Client(api_key=get_api_key())
    chat = client.chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(temperature=1.0),
    )

    print(f"[model] {MODEL}")
    try:
        question = args.question
        while True:
            print(f"[質問] {question}\n")
            print(ask(question, chat))
            try:
                question = input("\n続けて質問してください（Ctrl-Dで終了）: ").strip()
            except EOFError:
                print()
                break
            if not question:
                continue
            print()
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
