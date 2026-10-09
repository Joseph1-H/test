"""The `atlas` command: interactive chat, one-shot `ask`, and `doctor` checks."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from typing import TextIO

from . import __version__
from .config import ConfigError, Settings, load_settings
from .doctor import run_checks
from .logging_setup import setup_logging
from .providers import OllamaProvider, ProviderError
from .session import CloudNotAllowed, Session
from .ui import Style

log = logging.getLogger("atlas.cli")

HELP = """Commands:
  /help            Show this help
  /model           List models for the active provider
  /model NAME      Switch to model NAME
  /local           Use the local model (Ollama)
  /cloud           Use the cloud model (asks for permission first)
  /status          Show model, cloud state and workspace
  /doctor          Check GPU, Ollama and configuration
  /clear           Forget this conversation
  /history         Show this conversation
  /exit            Quit (also Ctrl+D)

Anything else is sent to the model. Press Ctrl+C to stop a reply."""


class AtlasCLI:
    def __init__(
        self,
        session: Session,
        input_fn: Callable[[str], str] = input,
        out: TextIO = sys.stdout,
        style: Style | None = None,
    ) -> None:
        self.session = session
        self.input = input_fn
        self.out = out
        self.s = style or Style(out)
        self.commands: dict[str, Callable[[str], bool]] = {
            "/help": self.cmd_help,
            "/model": self.cmd_model,
            "/local": self.cmd_local,
            "/cloud": self.cmd_cloud,
            "/status": self.cmd_status,
            "/doctor": self.cmd_doctor,
            "/clear": self.cmd_clear,
            "/history": self.cmd_history,
            "/exit": self.cmd_exit,
            "/quit": self.cmd_exit,
        }

    # ---- output helpers ----------------------------------------------------------
    def say(self, text: str = "") -> None:
        print(text, file=self.out, flush=True)

    def atlas(self, text: str) -> None:
        self.say(f"{self.s.cyan(self.s.bold('ATLAS >'))} {text}")

    def error(self, text: str) -> None:
        self.say(f"{self.s.red('Error:')} {text}")

    def confirm(self, prompt: str) -> bool:
        self.say(self.s.yellow(prompt))
        try:
            answer = self.input("[y/N] ")
        except (EOFError, KeyboardInterrupt):
            self.say()
            return False
        return answer.strip().lower() in ("y", "yes")

    def cloud_state(self) -> str:
        st = self.session.settings
        if not st.cloud_configured:
            return "off (not configured)"
        if not st.cloud_allowed:
            return "off (disabled in .env)"
        if self.session.cloud_authorized:
            return "authorized for this session"
        return "available (asks before use)"

    # ---- main loop -------------------------------------------------------------
    def banner(self) -> None:
        self.say(self.s.bold(f"ATLAS v{__version__}"))
        self.say(self.s.dim("Automated Tasks, Logic and Software"))
        self.say(f"Model: {self.session.active.label}")
        self.say(f"Cloud: {self.cloud_state()}")
        self.say(f"Workspace: {self.session.settings.workspace}")
        self.say(self.s.dim("Type /help for commands."))
        self.say()

    def run(self) -> int:
        self.banner()
        while True:
            try:
                line = self.input(f"{self.s.green(self.s.bold('You >'))} ")
            except EOFError:
                self.say()
                return 0
            except KeyboardInterrupt:
                self.say(self.s.dim("\n(Use /exit or Ctrl+D to quit.)"))
                continue
            if not self.handle(line):
                return 0

    def handle(self, line: str) -> bool:
        """Process one line of input. Returns False when the user wants to quit."""
        line = line.strip()
        if not line:
            return True
        if line.startswith("/"):
            name, _, arg = line.partition(" ")
            command = self.commands.get(name.lower())
            if command is None:
                self.error(f"Unknown command {name}. Type /help.")
                return True
            return command(arg.strip())
        self.chat(line)
        return True

    def chat(self, text: str) -> None:
        self.out.write(f"{self.s.cyan(self.s.bold('ATLAS >'))} ")
        self.out.flush()

        def on_token(piece: str) -> None:
            self.out.write(piece)
            self.out.flush()

        try:
            self.session.send(text, on_token=on_token)
            self.say()
        except KeyboardInterrupt:
            self.say(self.s.dim("\n[stopped; this reply was not saved]"))
        except (ProviderError, CloudNotAllowed) as err:
            self.say()
            self.error(str(err))

    # ---- commands ----------------------------------------------------------------
    def cmd_help(self, _: str) -> bool:
        self.say(HELP)
        return True

    def cmd_model(self, arg: str) -> bool:
        provider = self.session.active
        if not arg:
            try:
                models = provider.list_models()
            except ProviderError as err:
                self.error(str(err))
                return True
            if not models:
                self.say("No models available. For local models run: ollama pull <model>")
            for m in models:
                marker = self.s.green("*") if m == provider.model else " "
                self.say(f" {marker} {m}")
            return True
        if isinstance(provider, OllamaProvider):
            try:
                if not provider.has_model(arg):
                    self.error(f"{arg} is not downloaded. Run: ollama pull {arg}")
                    return True
            except ProviderError as err:
                self.error(str(err))
                return True
        self.session.set_model(arg)
        self.atlas(f"Now using {self.session.active.label}")
        return True

    def cmd_local(self, _: str) -> bool:
        self.session.use_local()
        self.atlas(f"Now using {self.session.active.label}")
        return True

    def cmd_cloud(self, _: str) -> bool:
        try:
            provider = self.session.enable_cloud(self.confirm)
        except (CloudNotAllowed, ProviderError) as err:
            self.error(str(err))
            return True
        self.atlas(f"Now using {provider.label}. Type /local to switch back.")
        return True

    def cmd_status(self, _: str) -> bool:
        st = self.session.settings
        self.say(f"ATLAS v{__version__}")
        self.say(f"Model:     {self.session.active.label}")
        self.say(f"Mode:      {self.session.mode}")
        self.say(f"Cloud:     {self.cloud_state()}")
        self.say(f"Workspace: {st.workspace}")
        self.say(f"History:   {len(self.session.history)} messages (this session only)")
        self.say(f"Log file:  {st.log_file}")
        return True

    def cmd_doctor(self, _: str) -> bool:
        print_checks(run_checks(self.session.settings), self.say, self.s)
        return True

    def cmd_clear(self, _: str) -> bool:
        self.session.clear()
        self.atlas("Conversation cleared.")
        return True

    def cmd_history(self, _: str) -> bool:
        if not self.session.history:
            self.say("No messages yet.")
            return True
        for msg in self.session.history:
            who = self.s.green("You") if msg.role == "user" else self.s.cyan("ATLAS")
            text = msg.content if len(msg.content) <= 300 else msg.content[:300] + " …"
            self.say(f"{who}: {text}")
        return True

    def cmd_exit(self, _: str) -> bool:
        self.say("Bye.")
        return False


def print_checks(checks, say: Callable[[str], None], s: Style) -> bool:
    """Print doctor results. Returns True if nothing failed."""
    icons = {"ok": s.green("✔"), "warn": s.yellow("!"), "fail": s.red("✘")}
    for c in checks:
        say(f" {icons[c.status]} {c.name:<17} {c.detail}")
    return not any(c.status == "fail" for c in checks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlas", description="ATLAS: a local-first AI coding assistant.")
    parser.add_argument("--version", action="version", version=f"ATLAS {__version__}")
    parser.add_argument("-w", "--workspace", help="project directory (default: current directory)")
    parser.add_argument("-m", "--model", help="local model to use (default: ATLAS_LOCAL_MODEL)")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("chat", help="interactive chat (default)")
    sub.add_parser("doctor", help="check GPU, Ollama, model and configuration")
    ask = sub.add_parser("ask", help="ask one question and print the answer")
    ask.add_argument("prompt", nargs="+", help="the question")
    ask.add_argument("--cloud", action="store_true",
                     help="use the cloud model (this flag is your explicit permission)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings: Settings = load_settings(workspace=args.workspace)
    except ConfigError as err:
        print(f"Configuration error: {err}", file=sys.stderr)
        return 2
    setup_logging(settings.log_file, settings.log_level)
    log.info("start", extra={"version": __version__, "command": args.command or "chat"})
    style = Style(sys.stdout)

    if args.command == "doctor":
        ok = print_checks(run_checks(settings), print, style)
        return 0 if ok else 1

    if not settings.workspace.is_dir():
        print(f"Workspace {settings.workspace} does not exist.", file=sys.stderr)
        return 2

    session = Session(settings)
    if args.model:
        session.set_model(args.model)

    if args.command == "ask":
        if args.cloud:
            try:
                session.enable_cloud(confirm=lambda _prompt: True)
            except (CloudNotAllowed, ProviderError) as err:
                print(f"Error: {err}", file=sys.stderr)
                return 2
        try:
            session.send(" ".join(args.prompt), on_token=lambda p: print(p, end="", flush=True))
            print()
            return 0
        except ProviderError as err:
            print(f"\nError: {err}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print()
            return 130

    try:
        import readline  # noqa: F401  # arrow keys and input history in the prompt
    except ImportError:
        pass
    return AtlasCLI(session, style=style).run()


if __name__ == "__main__":
    sys.exit(main())
