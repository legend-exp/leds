from __future__ import annotations

import argparse
import socket


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

    pn.serve(
        _bound_factory(args.base_path),
        address=args.address,
        port=args.port,
        websocket_origin=args.allow_websocket_origin or None,
        num_procs=args.num_procs,
        show=False,
    )


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
            nargs="?",
            default=None,
            help="production cycle directory (defaults to $LEDS_BASE_PATH)",
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
