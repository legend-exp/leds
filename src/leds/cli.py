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


def _prewarm(base_path):
    """Populate the shared caches before ``pn.serve`` forks its workers.

    Bokeh forks for ``--num-procs`` inside ``pn.serve``, so anything cached
    here is inherited copy-on-write by every worker and the first session each
    one serves is already warm. Best-effort: a bad path must still leave a
    server running that can report the problem in-app.

    Deliberately metadata and directory scans only -- **no HDF5 file is opened
    here**, because HDF5 is not fork-safe and an inherited handle would
    corrupt reads in the children.
    """
    from leds.config import discover_cycles, resolve_base_paths  # noqa: PLC0415
    from leds.event_viewer import EventViewer  # noqa: PLC0415

    try:
        cycles = discover_cycles(resolve_base_paths(base_path))
    except Exception as exc:
        print(  # noqa: T201 (operator-facing CLI warning)
            f"leds: pre-warm skipped ({type(exc).__name__}: {exc})", file=sys.stderr
        )
        return
    # one at a time, so a broken (e.g. half-written tmp) cycle does not leave
    # the others cold
    for label, path in cycles.items():
        try:
            viewer = EventViewer(path)
            runs = viewer.available_runs()
            # the newest run's channelmap is what a new session renders first
            for period in sorted(runs)[-1:]:
                for run in sorted(runs[period])[-1:]:
                    for tstamp in runs[period][run][:1]:
                        viewer._channelmap(tstamp)
                        viewer.statuses(tstamp)
        except Exception as exc:
            print(  # noqa: T201 (operator-facing CLI warning)
                f"leds: pre-warm of {label} skipped ({type(exc).__name__}: {exc})",
                file=sys.stderr,
            )


def _free_port():
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _cookie_secret(args):
    """The signing secret for auth cookies, warning when it is ephemeral."""
    import secrets  # noqa: PLC0415 (only needed when auth is enabled)

    if args.cookie_secret:
        return args.cookie_secret
    print(  # noqa: T201 (intentional operator-facing CLI warning)
        "leds: no --cookie-secret/$LEDS_COOKIE_SECRET set; using an "
        "ephemeral one. Provide a fixed secret so logins survive "
        "restarts and are shared across replicas.",
        file=sys.stderr,
    )
    return secrets.token_hex(32)


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

    # Optional login page, two flavours. LDAP ($LEDS_LDAP_*) takes precedence;
    # otherwise ``basic_auth`` is either a shared password or a path to a JSON
    # file of {username: password}. Secrets are typically injected by NERSC
    # Spin as env vars or mounted files.
    from leds.ldap_auth import LDAPConfig  # noqa: PLC0415 (lazy import)

    ldap_cfg = LDAPConfig.from_env()  # None unless $LEDS_LDAP_SERVER is set

    if ldap_cfg or args.basic_auth:
        import importlib.resources  # noqa: PLC0415 (only needed when auth is on)

        serve_kwargs["cookie_secret"] = _cookie_secret(args)
        # LEGEND-branded login/logout pages instead of Panel's defaults.
        templates = importlib.resources.files("leds") / "templates"
        login_template = str(templates / "login.html")
        logout_template = str(templates / "logout.html")

    if ldap_cfg:
        if args.basic_auth:
            print(  # noqa: T201 (intentional operator-facing CLI warning)
                "leds: LEDS_LDAP_SERVER is set; ignoring "
                "LEDS_BASIC_AUTH/--basic-auth.",
                file=sys.stderr,
            )
        from leds.ldap_auth import LDAPAuthProvider  # noqa: PLC0415

        # Note: must NOT also pass basic_auth/login_template to pn.serve, or
        # Panel would build its own auth provider and clobber this one.
        serve_kwargs["auth_provider"] = LDAPAuthProvider(
            ldap_cfg,
            login_template=login_template,
            logout_template=logout_template,
        )
    elif args.basic_auth:
        serve_kwargs["basic_auth"] = args.basic_auth
        serve_kwargs["login_template"] = login_template
        serve_kwargs["logout_template"] = logout_template

    if args.num_threads:
        # run callbacks on a thread pool so one session's blocking read does
        # not stall every other session in the worker; see "Threads" in the
        # README for what this does and does not buy. Set before pn.serve
        # forks: the pool spawns its threads lazily, so the children inherit
        # only the configuration, never a live thread.
        pn.config.nthreads = args.num_threads

    factory = _bound_factory(args.base_path)  # imports leds.app before forking
    if args.prewarm:
        _prewarm(args.base_path)
    pn.serve(factory, **serve_kwargs)


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
            help="one or more directories to search for production cycles, "
            "e.g. prod-blind/ref prod-blind/tmp prod-blind/auto (defaults to "
            "$LEDS_BASE_PATH, which may list several separated by the path "
            "separator)",
        )

    serve = sub.add_parser("serve", help="run the hosted multi-user server")
    add_base_path(serve)
    serve.add_argument("--address", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=5006)
    serve.add_argument("--num-procs", type=int, default=1)
    serve.add_argument(
        "--num-threads",
        type=int,
        default=0,
        help="size of the thread pool callbacks run on, so one session's slow "
        "read does not freeze the others in its process; 0 (the default) "
        "keeps everything on the event loop. The container passes "
        "$NUM_THREADS here, as it does $NUM_PROCS to --num-procs",
    )
    serve.add_argument(
        "--prewarm",
        action="store_true",
        default=os.environ.get("LEDS_PREWARM", "").lower() in ("1", "true", "yes"),
        help="scan the cycles and build the newest channelmap before serving, "
        "so the first session of every worker is warm (default $LEDS_PREWARM). "
        "Delays the listening socket by that much -- check it against any "
        "readiness probe",
    )
    serve.add_argument(
        "--allow-websocket-origin",
        action="append",
        help="host[:port] allowed to connect (repeatable)",
    )
    serve.add_argument(
        "--basic-auth",
        default=os.environ.get("LEDS_BASIC_AUTH"),
        help="enable a login page; a shared password or a path to a JSON file "
        "of {username: password} (default $LEDS_BASIC_AUTH). Ignored when "
        "LDAP login is configured via $LEDS_LDAP_SERVER",
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
