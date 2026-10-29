#!/usr/bin/env python3
"""Make one text-only context question; the expected code occurs only in its header."""
import argparse
import json
from pathlib import Path


def request_body(answer="VELM42", model="mf-direct", filler_lines=72):
    if not answer or not answer.isascii() or not answer.isalnum() or len(answer) > 16:
        raise ValueError("Use one short ASCII alphanumeric identifier")
    system = "Read the document, remember its identifier, and answer the final question with only that identifier."
    header = "Document header\nIdentifier: " + answer + "\n\nDocument entries:\n"
    filler = "\n".join("Entry " + str(i).zfill(3) +
        ": pale paper rests beside quiet stone while three small birds watch."
        for i in range(filler_lines))
    tail = "\nEnd of entries. What identifier was stated in the document header? Reply with that identifier only."
    body = dict(model=model, temperature=0, max_tokens=12,
                chat_template_kwargs={"enable_thinking": False},
                messages=[dict(role="system", content=system), dict(role="user", content=header + filler + tail)])
    if answer in system or answer in filler or answer in tail:
        raise ValueError("Identifier accidentally appears outside the header")
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answer", default="VELM42")
    parser.add_argument("--model", default="mf-direct")
    parser.add_argument("--filler-lines", type=int, default=72,
                        help="Seed targets roughly 1200-1600 tokens; actual /tokenize and slot checks decide")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.filler_lines < 1:
        parser.error("filler-lines must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(request_body(args.answer, args.model, args.filler_lines), stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
