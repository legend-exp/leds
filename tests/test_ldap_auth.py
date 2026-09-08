from __future__ import annotations

import argparse
from typing import ClassVar

import ldap3
import panel as pn
import pytest
from ldap3.core.exceptions import LDAPException
from panel.auth import LOGOUT_TEMPLATE, BasicLoginHandler

from leds.app import user_chip
from leds.cli import _serve
from leds.ldap_auth import (
    LOGIN_HINT,
    LDAPAuthProvider,
    LDAPConfig,
    LDAPLoginHandler,
)

DIRECT_ENV = {
    "LEDS_LDAP_SERVER": "ldaps://ldap.example:636",
    "LEDS_LDAP_USER_DN_TEMPLATE": "uid={username},ou=people,dc=example,dc=org",
}
SEARCH_ENV = {
    "LEDS_LDAP_SERVER": "ldaps://ldap.example:636",
    "LEDS_LDAP_BIND_DN": "cn=svc,dc=example,dc=org",
    "LEDS_LDAP_BIND_PASSWORD": "svcpw",
    "LEDS_LDAP_SEARCH_BASE": "ou=people,dc=example,dc=org",
}


# ---------------------------------------------------------------- config


def test_from_env_disabled_without_server():
    assert LDAPConfig.from_env({}) is None
    assert LDAPConfig.from_env({"LEDS_LDAP_USER_DN_TEMPLATE": "uid={username}"}) is None


def test_from_env_direct_bind():
    cfg = LDAPConfig.from_env(DIRECT_ENV)
    assert cfg.user_dn_template.startswith("uid={username}")
    assert cfg.bind_dn is None


def test_from_env_search_bind():
    cfg = LDAPConfig.from_env(SEARCH_ENV)
    assert cfg.user_dn_template is None
    assert cfg.search_base == "ou=people,dc=example,dc=org"


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"LEDS_LDAP_USER_DN_TEMPLATE": "uid=fixed,ou=people"}, "username"),
        ({"LEDS_LDAP_SERVER": "ldap://x", "LEDS_LDAP_STARTTLS": "0"}, "incomplete"),
        (
            {**SEARCH_ENV, "LEDS_LDAP_SERVER": "ldap://x", "LEDS_LDAP_BIND_DN": ""},
            "LEDS_LDAP_BIND_DN",
        ),
        ({**DIRECT_ENV, "LEDS_LDAP_STARTTLS": "1"}, "ldaps"),
        ({**SEARCH_ENV, "LEDS_LDAP_USER_FILTER": "(uid=fixed)"}, "username"),
    ],
)
def test_from_env_invalid(overrides, match):
    env = {"LEDS_LDAP_SERVER": "ldaps://ldap.example:636", **overrides}
    with pytest.raises(ValueError, match=match):
        LDAPConfig.from_env(env)


# ---------------------------------------------------------------- fakes


class FakeConnection:
    """Stands in for ldap3.Connection; behaviour driven by class attrs."""

    passwords: ClassVar[dict] = {}  # dn -> password accepted by bind()
    start_tls_ok: ClassVar[bool] = True
    search_results: ClassVar[dict] = {}  # search base -> list of entry DNs
    instances: ClassVar[list] = []

    def __init__(self, server, user=None, password=None, **kwargs):
        self.server, self.user, self.password = server, user, password
        self.kwargs = kwargs
        self.entries = []
        self.bound = False
        self.unbound = False
        self.started_tls = False
        self.searches = []
        FakeConnection.instances.append(self)

    def start_tls(self):
        self.started_tls = True
        return self.start_tls_ok

    def bind(self):
        self.bound = self.passwords.get(self.user) == self.password
        return self.bound

    def unbind(self):
        self.unbound = True

    def search(self, base, filt, **kwargs):
        self.searches.append((base, filt, kwargs))
        self.entries = [FakeEntry(dn) for dn in self.search_results.get(base, [])]
        return bool(self.entries)


class FakeEntry:
    def __init__(self, dn):
        self.entry_dn = dn


@pytest.fixture
def fake_ldap(monkeypatch):
    FakeConnection.passwords = {}
    FakeConnection.start_tls_ok = True
    FakeConnection.search_results = {}
    FakeConnection.instances = []
    monkeypatch.setattr(ldap3, "Connection", FakeConnection)
    monkeypatch.setattr(ldap3, "Server", lambda url, **kwargs: url)
    monkeypatch.setattr(ldap3, "Tls", lambda **kwargs: None)
    return FakeConnection


def make_handler(env):
    """A handler with only the attributes _validate needs (no tornado app)."""
    handler = LDAPLoginHandler.__new__(LDAPLoginHandler)
    handler._ldap_config = LDAPConfig.from_env(env)
    return handler


# ---------------------------------------------------------------- _validate


@pytest.mark.parametrize(
    ("username", "password"), [("", "pw"), ("  ", "pw"), ("alice", ""), ("", "")]
)
def test_empty_credentials_rejected_without_network(fake_ldap, username, password):
    # An empty password must never reach the server: it would be an LDAP
    # unauthenticated bind, which many servers report as success.
    handler = make_handler(DIRECT_ENV)
    assert handler._validate(username, password) is False
    assert fake_ldap.instances == []


def test_direct_bind_success_and_dn_escaping(fake_ldap):
    dn = "uid=alice\\,ou\\=x,ou=people,dc=example,dc=org"
    fake_ldap.passwords = {dn: "pw"}
    handler = make_handler(DIRECT_ENV)
    # the comma in the username must be escaped into the DN, not split it
    assert handler._validate("alice,ou=x", "pw") is True
    (conn,) = fake_ldap.instances
    assert conn.user == dn
    assert conn.unbound


def test_direct_bind_bad_password(fake_ldap):
    fake_ldap.passwords = {"uid=alice,ou=people,dc=example,dc=org": "right"}
    handler = make_handler(DIRECT_ENV)
    assert handler._validate("alice", "wrong") is False
    assert handler._auth_error is None  # generic "invalid username or password"


def test_search_bind_success_and_filter_escaping(fake_ldap):
    user_dn = "uid=alice,ou=people,dc=example,dc=org"
    fake_ldap.passwords = {"cn=svc,dc=example,dc=org": "svcpw", user_dn: "pw"}
    fake_ldap.search_results = {"ou=people,dc=example,dc=org": [user_dn]}
    handler = make_handler(SEARCH_ENV)

    assert handler._validate("alice", "pw") is True
    svc, user = fake_ldap.instances
    assert svc.unbound
    assert user.user == user_dn

    # a filter-injection attempt reaches the server fully escaped
    fake_ldap.instances.clear()
    fake_ldap.search_results = {}  # nothing matches the (escaped) filter
    assert handler._validate("*)(uid=*", "pw") is False
    (svc,) = fake_ldap.instances
    assert svc.searches[0][1] == "(uid=\\2a\\29\\28uid=\\2a)"


@pytest.mark.parametrize("hits", [[], ["uid=a,ou=p", "uid=b,ou=p"]])
def test_search_bind_requires_exactly_one_entry(fake_ldap, hits):
    fake_ldap.passwords = {"cn=svc,dc=example,dc=org": "svcpw"}
    fake_ldap.search_results = {"ou=people,dc=example,dc=org": hits}
    handler = make_handler(SEARCH_ENV)
    assert handler._validate("alice", "pw") is False


def test_search_bind_bad_service_account_is_unavailable(fake_ldap, capsys):
    fake_ldap.passwords = {}  # service bind fails
    handler = make_handler(SEARCH_ENV)
    assert handler._validate("alice", "pw") is False
    assert handler._auth_error == LDAPLoginHandler._AUTH_UNAVAILABLE
    assert "LDAP error" in capsys.readouterr().err


# ---------------------------------------------------------------- group check


def test_group_membership_required(fake_ldap):
    env = {**DIRECT_ENV, "LEDS_LDAP_GROUP_DN": "cn=leds,ou=groups,dc=example,dc=org"}
    user_dn = "uid=alice,ou=people,dc=example,dc=org"
    fake_ldap.passwords = {user_dn: "pw"}

    # not a member: the group entry does not match the membership filter
    handler = make_handler(env)
    assert handler._validate("alice", "pw") is False
    (conn,) = fake_ldap.instances
    base, filt, kwargs = conn.searches[0]
    assert base == "cn=leds,ou=groups,dc=example,dc=org"
    for atom in (f"member={user_dn}", f"uniqueMember={user_dn}", "memberUid=alice"):
        assert atom in filt
    assert kwargs["search_scope"] == ldap3.BASE

    # member: same search now yields the group entry
    fake_ldap.instances.clear()
    fake_ldap.search_results = {env["LEDS_LDAP_GROUP_DN"]: [env["LEDS_LDAP_GROUP_DN"]]}
    assert handler._validate("alice", "pw") is True


def test_no_group_configured_skips_search(fake_ldap):
    fake_ldap.passwords = {"uid=alice,ou=people,dc=example,dc=org": "pw"}
    handler = make_handler(DIRECT_ENV)
    assert handler._validate("alice", "pw") is True
    (conn,) = fake_ldap.instances
    assert conn.searches == []


# ---------------------------------------------------------------- errors


def test_failed_starttls_never_reaches_bind(fake_ldap, capsys):
    # a failed StartTLS negotiation must abort, not bind over plaintext
    fake_ldap.passwords = {"uid=alice,ou=people,dc=example,dc=org": "pw"}
    fake_ldap.start_tls_ok = False
    env = {
        **DIRECT_ENV,
        "LEDS_LDAP_SERVER": "ldap://ldap.example:389",
        "LEDS_LDAP_STARTTLS": "1",
    }
    handler = make_handler(env)
    assert handler._validate("alice", "pw") is False
    assert handler._auth_error == LDAPLoginHandler._AUTH_UNAVAILABLE
    assert "StartTLS" in capsys.readouterr().err
    assert not any(c.bound for c in fake_ldap.instances)


@pytest.mark.usefixtures("fake_ldap")
def test_server_error_reports_unavailable(monkeypatch, capsys):
    def boom(*args, **kwargs):
        msg = "socket connection error"
        raise LDAPException(msg)

    monkeypatch.setattr(ldap3, "Connection", boom)
    handler = make_handler(DIRECT_ENV)
    assert handler._validate("alice", "pw") is False
    assert handler._auth_error == LDAPLoginHandler._AUTH_UNAVAILABLE
    err = capsys.readouterr().err
    assert "LDAP error" in err
    assert "pw" not in err  # never log the password


# ---------------------------------------------------------------- CLI wiring


def serve_args(**overrides):
    defaults = {
        "address": "0.0.0.0",
        "port": 5006,
        "num_procs": 1,
        "allow_websocket_origin": None,
        "basic_auth": None,
        "cookie_secret": "s3cret",
        "base_path": None,
        "prewarm": False,
        "num_threads": 0,
    }
    return argparse.Namespace(**{**defaults, **overrides})


@pytest.fixture
def serve_kwargs(monkeypatch):
    captured = {}

    def fake_serve(_factory, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(pn, "serve", fake_serve)
    return captured


def test_cli_ldap_takes_precedence(serve_kwargs, monkeypatch, capsys):
    for key, value in DIRECT_ENV.items():
        monkeypatch.setenv(key, value)
    _serve(serve_args(basic_auth="hunter2"))

    assert isinstance(serve_kwargs["auth_provider"], LDAPAuthProvider)
    assert "basic_auth" not in serve_kwargs
    assert "login_template" not in serve_kwargs
    assert serve_kwargs["cookie_secret"] == "s3cret"
    assert "ignoring" in capsys.readouterr().err


def test_cli_basic_auth_unchanged(serve_kwargs, monkeypatch):
    monkeypatch.delenv("LEDS_LDAP_SERVER", raising=False)
    _serve(serve_args(basic_auth="hunter2"))

    assert serve_kwargs["basic_auth"] == "hunter2"
    assert serve_kwargs["login_template"].endswith("login.html")
    assert "auth_provider" not in serve_kwargs


def test_cli_no_auth(serve_kwargs, monkeypatch):
    monkeypatch.delenv("LEDS_LDAP_SERVER", raising=False)
    _serve(serve_args())

    assert "auth_provider" not in serve_kwargs
    assert "basic_auth" not in serve_kwargs
    assert "cookie_secret" not in serve_kwargs


@pytest.mark.usefixtures("serve_kwargs")
def test_cli_num_threads_enables_the_thread_pool(monkeypatch):
    monkeypatch.delenv("LEDS_LDAP_SERVER", raising=False)
    monkeypatch.setattr(pn.config, "nthreads", None)
    _serve(serve_args(num_threads=3))

    assert pn.config.nthreads == 3


@pytest.mark.usefixtures("serve_kwargs")
def test_cli_threads_off_by_default(monkeypatch):
    monkeypatch.delenv("LEDS_LDAP_SERVER", raising=False)
    monkeypatch.setattr(pn.config, "nthreads", None)
    _serve(serve_args())

    assert pn.config.nthreads is None


# ---------------------------------------------------------------- logout/header


def test_branded_logout_template_wired(serve_kwargs, monkeypatch):
    for key, value in DIRECT_ENV.items():
        monkeypatch.setenv(key, value)
    _serve(serve_args())
    provider = serve_kwargs["auth_provider"]
    assert provider._logout_template is not LOGOUT_TEMPLATE  # branded, not stock
    html = provider._logout_template.render(PANEL_CDN="", LOGIN_ENDPOINT="/login")
    assert "signed out" in html
    assert 'action="./login"' in html  # the way back in


def test_login_hint_shown_only_in_ldap_mode(serve_kwargs, monkeypatch):
    for key, value in DIRECT_ENV.items():
        monkeypatch.setenv(key, value)
    _serve(serve_args())
    template = serve_kwargs["auth_provider"]._login_template

    # LDAP mode: the handler passes the hint, so the page carries it
    html = template.render(
        login_endpoint="/login",
        errormessage="",
        login_hint=LOGIN_HINT,
        PANEL_CDN="",
    )
    assert LOGIN_HINT in html
    assert '<p class="login-hint">' in html

    # shared-password mode renders the same template without the variable
    # (the .login-hint CSS rule is always present; the element is not)
    plain = template.render(login_endpoint="/login", errormessage="", PANEL_CDN="")
    assert '<p class="login-hint">' not in plain
    assert LOGIN_HINT not in plain


def test_ldap_handler_overrides_get():
    # the hint only reaches the page through our get(); guard the override
    assert LDAPLoginHandler.get is not BasicLoginHandler.get


def test_cli_basic_auth_logout_template(serve_kwargs, monkeypatch):
    monkeypatch.delenv("LEDS_LDAP_SERVER", raising=False)
    _serve(serve_args(basic_auth="hunter2"))
    assert serve_kwargs["logout_template"].endswith("logout.html")


def test_user_chip():
    chip = user_chip("ggmarshall")
    assert "ggmarshall" in chip
    assert 'href="./logout"' in chip
    assert user_chip("<b>x</b>") is not None
    assert "<b>" not in user_chip("<b>x</b>")  # escaped
    assert user_chip(None) is None
    assert user_chip("guest") is None
