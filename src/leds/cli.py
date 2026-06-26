from __future__ import annotations

import argparse
import os
import socket
import sys


def _bound_factory(base_path):
    """Return a zero-arg ``create_app`` bound to ``base_path`` (per-session).

    A plain closure (not ``functools.partial``) so Panel recognises it as a
    session factory and calls it per connection instead of rendering its repr.
    """
    from leds.app import create_app  # noqa: PLC0415 (lazy: keep panel off startup)

    def factory():
        return create_app(base_path)

    return factory


def _free_port():
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _serve(args):
    """Long-running, multi-user hosted instance (Docker / NERSC spin)."""
    import panel as pn  # noqa: PLC0415 (lazy: keep panel off CLI startup)

    serve_kwargs = {
        "address": args.address,
        "port": args.port,
        "websocket_origin": args.allow_websocket_origin or None,
        "num_procs": args.num_procs,
        "show": False,
    }

    # Optional login page. ``basic_auth`` is either a shared password or a path
    # to a JSON file of {username: password}; both are typically injected by
    # NERSC Spin as a secret (env var or mounted file).
    if args.basic_auth:
        import secrets  # noqa: PLC0415 (only needed when auth is enabled)

        cookie_secret = args.cookie_secret or secrets.token_hex(32)
        if not args.cookie_secret:
            print(  # noqa: T201 (intentional operator-facing CLI warning)
                "leds: no --cookie-secret/$LEDS_COOKIE_SECRET set; using an "
                "ephemeral one. Provide a fixed secret so logins survive "
                "restarts and are shared across replicas.",
                file=sys.stderr,
            )
        serve_kwargs["basic_auth"] = args.basic_auth
        serve_kwargs["cookie_secret"] = cookie_secret

    pn.serve(_bound_factory(args.base_path), **serve_kwargs)


def _app(args):
    """Local single-user instance: a browser tab, or a native window."""
    import panel as pn  # noqa: PLC0415 (lazy: keep panel off CLI startup)

    factory = _bound_factory(args.base_path)

    if args.desktop:
        import webview  # noqa: PLC0415 (optional dep, imported only when needed)

        port = args.port or _free_port()
        pn.serve(factory, port=port, show=False, threaded=True)
        webview.create_window(
            "LEGEND Event Display",
            f"http://localhost:{port}",
            width=1400,
            height=900,
        )
        webview.start()
    else:
        pn.serve(factory, port=args.port, show=True)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="leds", description="LEGEND event display")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_base_path(p):
        p.add_argument(
            "base_path",
            nargs="*",
            default=None,
            help="one or more directories to search for production cycles "
            "(defaults to $LEDS_BASE_PATH, which may list several separated "
            "by the path separator)",
        )

    serve = sub.add_parser("serve", help="run the hosted multi-user server")
    add_base_path(serve)
    serve.add_argument("--address", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=5006)
    serve.add_argument("--num-procs", type=int, default=1)
    serve.add_argument(
        "--allow-websocket-origin",
        action="append",
        help="host[:port] allowed to connect (repeatable)",
    )
    serve.add_argument(
        "--basic-auth",
        default=os.environ.get("LEDS_BASIC_AUTH"),
        help="enable a login page; a shared password or a path to a JSON file "
        "of {username: password} (default $LEDS_BASIC_AUTH)",
    )
    serve.add_argument(
        "--cookie-secret",
        default=os.environ.get("LEDS_COOKIE_SECRET"),
        help="secret used to sign the auth cookie; use a fixed value across "
        "restarts/replicas (default $LEDS_COOKIE_SECRET)",
    )
    serve.set_defaults(func=_serve)

    app = sub.add_parser("app", help="run a local single-user instance")
    add_base_path(app)
    app.add_argument("--port", type=int, default=0, help="0 picks a free port")
    app.add_argument(
        "--desktop",
        action="store_true",
        help="open in a native window (requires leds[desktop])",
    )
    app.set_defaults(func=_app)

    args = parser.parse_args(argv)
    args.func(args)
