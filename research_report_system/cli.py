"""CLI: python -m research_report_system.cli plan "<topic>" """

import argparse
import sys

from research_report_system.llm import StructuredOutputError
from research_report_system.planner import plan_topic


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research_report_system.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", help="Generate a research plan for a topic")
    p.add_argument("topic")
    args = parser.parse_args(argv)

    try:
        plan = plan_topic(args.topic)
    except StructuredOutputError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # e.g. Ollama unreachable
        print(f"ERROR: {exc!r}", file=sys.stderr)
        return 2
    print(plan.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
