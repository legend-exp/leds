"""LDAP authentication for the hosted server (``leds serve``).

Activated by setting ``$LEDS_LDAP_SERVER``; see :meth:`LDAPConfig.from_env`
for the full set of environment variables. Two bind strategies:

- *direct bind*: ``$LEDS_LDAP_USER_DN_TEMPLATE`` turns the login name into a
  DN and the user's own credentials are tried directly;
- *search-then-bind*: a service account (``$LEDS_LDAP_BIND_DN`` /
  ``$LEDS_LDAP_BIND_PASSWORD``) looks the user up under
  ``$LEDS_LDAP_SEARCH_BASE`` and the found DN is re-bound with the user's
  password.

Optionally ``$LEDS_LDAP_GROUP_DN`` restricts access to members of one group.

Plugs into Panel's basic-auth machinery by subclassing the (semi-internal)
``BasicLoginHandler``/``BasicAuthProvider`` pair — the same extension pattern
Panel itself uses for PAM auth. Known-good on Panel 1.9.x; re-check the two
overridden surfaces (``_validate``/``post`` and the ``login_handler``
property) when upgrading Panel.
"""

from __future__ import annotations

import dataclasses
import os
import ssl
import sys

import certifi
import ldap3
import tornado.escape
from ldap3.core.exceptions import LDAPException
from ldap3.utils.conv import escape_filter_chars
from ldap3.utils.dn import escape_rdn
from panel.auth import BasicAuthProvider, BasicLoginHandler
from panel.io.state import state

_TIMEOUT = 5  # seconds; _validate blocks the IOLoop, so keep LDAP calls short

#: Shown on the login page in LDAP mode only. Assumes the default
#: ``(uid={username})`` user filter; reword if a deployment ever matches users
#: on their mail attribute instead.
LOGIN_HINT = "Use your LEGEND LDAP credentials (username, not email)."


@dataclasses.dataclass(frozen=True)
class LDAPConfig:
    """LDAP connection settings, normally read from the environment."""

    server_url: str
    user_dn_template: str | None = None
    bind_dn: str | None = None
    bind_password: str | None = None
    search_base: str | None = None
    user_filter: str = "(uid={username})"
    group_dn: str | None = None
    starttls: bool = False
    ca_file: str | None = None

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> LDAPConfig | None:
        """Build a config from ``$LEDS_LDAP_*``, or ``None`` if LDAP is off.

        Raises ``ValueError`` on an incomplete or contradictory configuration
        so a misconfigured server fails at startup, not at first login.
        """
        if env is None:
            env = dict(os.environ)
        server_url = env.get("LEDS_LDAP_SERVER")
        if not server_url:
            return None
        cfg = cls(
            server_url=server_url,
            user_dn_template=env.get("LEDS_LDAP_USER_DN_TEMPLATE") or None,
            bind_dn=env.get("LEDS_LDAP_BIND_DN") or None,
            bind_password=env.get("LEDS_LDAP_BIND_PASSWORD") or None,
            search_base=env.get("LEDS_LDAP_SEARCH_BASE") or None,
            user_filter=env.get("LEDS_LDAP_USER_FILTER") or "(uid={username})",
            group_dn=env.get("LEDS_LDAP_GROUP_DN") or None,
            starttls=env.get("LEDS_LDAP_STARTTLS", "").lower() in ("1", "true", "yes"),
            ca_file=env.get("LEDS_LDAP_CA_FILE") or None,
        )
        cfg._check()
        return cfg

    def _check(self) -> None:
        if self.starttls and self.server_url.lower().startswith("ldaps://"):
            msg = (
                "LEDS_LDAP_STARTTLS only applies to ldap:// URLs; "
                "ldaps:// is already TLS"
            )
            raise ValueError(msg)
        if self.user_dn_template is not None:
            if "{username}" not in self.user_dn_template:
                msg = "LEDS_LDAP_USER_DN_TEMPLATE must contain '{username}'"
                raise ValueError(msg)
            return  # direct-bind mode; service account not required
        missing = [
            name
            for name, value in (
                ("LEDS_LDAP_BIND_DN", self.bind_dn),
                ("LEDS_LDAP_BIND_PASSWORD", self.bind_password),
                ("LEDS_LDAP_SEARCH_BASE", self.search_base),
            )
            if not value
        ]
        if missing:
            msg = (
                "incomplete LDAP configuration: set LEDS_LDAP_USER_DN_TEMPLATE "
                "(direct bind) or all of LEDS_LDAP_BIND_DN, "
                "LEDS_LDAP_BIND_PASSWORD and LEDS_LDAP_SEARCH_BASE "
                f"(search-then-bind); missing {', '.join(missing)}"
            )
            raise ValueError(msg)
        if "{username}" not in self.user_filter:
            msg = "LEDS_LDAP_USER_FILTER must contain '{username}'"
            raise ValueError(msg)


class LDAPLoginHandler(BasicLoginHandler):
    """Validates the login form against an LDAP directory."""

    _ldap_config: LDAPConfig  # set by LDAPAuthProvider.login_handler

    _AUTH_UNAVAILABLE = "Authentication service unavailable; try again later."

    def _validate(self, username: str, password: str) -> bool:
        self._auth_error: str | None = None
        # An empty password would be an LDAP *unauthenticated bind*, which
        # many servers report as success. Reject before touching the network.
        if not username or not username.strip() or not password:
            return False
        try:
            return self._ldap_check(self._ldap_config, username, password)
        except LDAPException as exc:
            # Server unreachable, TLS failure, service-account rejected, ...
            # Details go to the operator only; the user gets a generic notice.
            print(  # noqa: T201 (intentional operator-facing log)
                f"leds: LDAP error during login for {username!r}: {exc!r}",
                file=sys.stderr,
            )
            self._auth_error = self._AUTH_UNAVAILABLE
            return False

    def get(self) -> None:
        # Copy of BasicLoginHandler.get (Panel 1.9.x) plus ``login_hint``, so
        # the page can say which credentials to use. The shared-password mode
        # renders the same template without it and shows no hint.
        from panel.auth import _validate_next_url  # noqa: PLC0415
        from panel.io.resources import CDN_DIST  # noqa: PLC0415

        try:
            errormessage = self.get_argument("error")
        except Exception:
            errormessage = ""
        next_url = _validate_next_url(self.get_argument("next", state.base_url))
        if next_url:
            self.set_cookie("next_url", next_url)
        self.write(
            self._login_template.render(
                login_endpoint=self._login_endpoint,
                errormessage=errormessage,
                login_hint=LOGIN_HINT,
                PANEL_CDN=CDN_DIST,
            )
        )

    def post(self) -> None:
        # Copy of BasicLoginHandler.post (Panel 1.9.x) except the error
        # message, so infrastructure failures read differently from bad
        # credentials without leaking which one of user/password/group failed.
        from panel.auth import _validate_next_url  # noqa: PLC0415

        username = self.get_argument("username", "")
        password = self.get_argument("password", "")
        if self._validate(username, password):
            self.set_current_user(username)
            next_url = _validate_next_url(self.get_cookie("next_url", state.base_url))
            self.redirect(next_url)
        else:
            error = self._auth_error or "Invalid username or password!"
            self.redirect(
                self.request.uri + "?error=" + tornado.escape.url_escape(error)
            )

    def _ldap_check(self, cfg: LDAPConfig, username: str, password: str) -> bool:
        server = self._server(cfg)

        if cfg.user_dn_template:
            # Direct bind: derive the DN and try the user's own credentials.
            user_dn = cfg.user_dn_template.format(username=escape_rdn(username))
            conn = self._connect(server, cfg, user_dn, password)
            if conn is None:
                return False
            ok = self._in_group(conn, cfg, user_dn, username)
            conn.unbind()
            return ok

        # Search-then-bind: locate the entry as the service account, then
        # verify the password by binding as the found DN.
        svc = self._connect(server, cfg, cfg.bind_dn, cfg.bind_password)
        if svc is None:
            msg = "LDAP service-account bind failed (check LEDS_LDAP_BIND_DN/_PASSWORD)"
            raise LDAPException(msg)
        try:
            filt = cfg.user_filter.format(username=escape_filter_chars(username))
            svc.search(cfg.search_base, filt, attributes=[])
            if len(svc.entries) != 1:  # unknown or ambiguous user
                return False
            user_dn = svc.entries[0].entry_dn
            if not self._in_group(svc, cfg, user_dn, username):
                return False
        finally:
            svc.unbind()
        conn = self._connect(server, cfg, user_dn, password)
        if conn is None:
            return False
        conn.unbind()
        return True

    @staticmethod
    def _server(cfg: LDAPConfig) -> ldap3.Server:
        tls = ldap3.Tls(
            validate=ssl.CERT_REQUIRED,
            ca_certs_file=cfg.ca_file or certifi.where(),
        )
        return ldap3.Server(cfg.server_url, tls=tls, connect_timeout=_TIMEOUT)

    @staticmethod
    def _connect(
        server: ldap3.Server, cfg: LDAPConfig, user_dn: str | None, password: str | None
    ) -> ldap3.Connection | None:
        """Bind as ``user_dn``; ``None`` means the credentials were rejected."""
        conn = ldap3.Connection(
            server, user=user_dn, password=password, receive_timeout=_TIMEOUT
        )
        if cfg.starttls and not conn.start_tls():
            # never fall through to a plaintext bind on a failed negotiation
            conn.unbind()
            msg = "LDAP StartTLS negotiation failed"
            raise LDAPException(msg)
        if not conn.bind():
            conn.unbind()
            return None
        return conn

    @staticmethod
    def _in_group(
        conn: ldap3.Connection, cfg: LDAPConfig, user_dn: str, username: str
    ) -> bool:
        """Check group membership, if a group is configured at all.

        One BASE-scope read of the group entry, matching groupOfNames
        (``member``), groupOfUniqueNames (``uniqueMember``) and posixGroup
        (``memberUid``) in a single filter.
        """
        if not cfg.group_dn:
            return True
        dn, uid = escape_filter_chars(user_dn), escape_filter_chars(username)
        filt = f"(|(member={dn})(uniqueMember={dn})(memberUid={uid}))"
        conn.search(cfg.group_dn, filt, search_scope=ldap3.BASE, attributes=[])
        return bool(conn.entries)


class LDAPAuthProvider(BasicAuthProvider):
    """``BasicAuthProvider`` whose login form checks LDAP, not a password list."""

    def __init__(self, ldap_config: LDAPConfig, **kwargs: object):
        self._ldap_config = ldap_config
        super().__init__(**kwargs)

    @property
    def login_handler(self) -> type[LDAPLoginHandler]:
        # Same class-attribute wiring as the base property (one provider per
        # process), just targeting our handler subclass.
        LDAPLoginHandler._login_endpoint = self._login_endpoint
        LDAPLoginHandler._login_template = self._login_template
        LDAPLoginHandler._ldap_config = self._ldap_config
        return LDAPLoginHandler
