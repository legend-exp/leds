# leds

![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/legend-exp/leds?logo=git)
[![GitHub Workflow Status](https://img.shields.io/github/checks-status/legend-exp/leds/main?label=main%20branch&logo=github)](https://github.com/legend-exp/leds/actions)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Codecov](https://img.shields.io/codecov/c/github/legend-exp/leds?logo=codecov)](https://app.codecov.io/gh/legend-exp/leds)
![GitHub issues](https://img.shields.io/github/issues/legend-exp/leds?logo=github)
![GitHub pull requests](https://img.shields.io/github/issues-pr/legend-exp/leds?logo=github)
![License](https://img.shields.io/github/license/legend-exp/leds)
[![Read the Docs](https://img.shields.io/readthedocs/leds?logo=readthedocs)](https://leds.readthedocs.io)

An event viewer for the LEGEND experiment: a [Panel](https://panel.holoviz.org)
dashboard with event display, waveforms, spectra, dataset-metadata and
validation views over a legend-dataflow production cycle.

## Running

```bash
pip install legend-leds   # or: uv pip install .

leds app /path/to/production_cycle          # local single-user browser tab
leds app --desktop /path/to/cycle           # native window (needs leds[desktop])
leds serve /path/to/cycle --port 5006       # hosted multi-user server
```

The production-cycle path (a directory containing `dataflow-config.yaml`, or a
directory of such cycles) can also come from `$LEDS_BASE_PATH`; several paths
may be listed separated by `:`.

Cycles inside directories named `ref`, `tmp` or `auto` are listed in that order
as sections of the production-cycle dropdown:

```bash
export LEDS_BASE_PATH=$PRODENV/prod-blind/ref:$PRODENV/prod-blind/tmp:$PRODENV/prod-blind/auto
```

Within a section, symlinks (`auto/latest`, `tmp/p19+`) come first, then the
cycles themselves, each newest first by creation time. The first entry is
selected at start. Cycles without a `pet` or `evt` tier (e.g. raw-only ones) are
not listed, since there are no events to show.

## Deployment (NERSC Spin)

The `Dockerfile` builds a slim two-stage image by cloning this repository:

```bash
docker build -t registry.nersc.gov/<project>/leds:latest \
  --build-arg LEDS_REPO=https://github.com/legend-exp/leds.git \
  --build-arg LEDS_REF=main .
docker push registry.nersc.gov/<project>/leds:latest
```

(Point `LEDS_REPO` at a fork while changes are not upstream yet.) The container
listens on port 5006 and runs under an arbitrary non-root UID, as Spin requires.

Mount the whole `prod-blind` directory as **one** read-only volume, e.g. at
`/prod-blind`, and set
`LEDS_BASE_PATH=/prod-blind/ref:/prod-blind/tmp:/prod-blind/auto`. Two reasons:

- A cycle's section comes from the name of its parent directory **inside the
  container**, so the mount points must end in `ref`, `tmp` and `auto`.
- tmp and auto cycles read their raw tier through relative paths into ref
  (`$_/../../ref/v3.0.0/generated/tier/raw`), so separate volumes break their
  waveforms.

### Service environment

| Variable                          | Required          | Meaning                                                                                                                                             |
| --------------------------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `LEDS_BASE_PATH`                  | yes               | production cycle(s) on the mounted filesystem (e.g. CFS)                                                                                            |
| `BOKEH_ALLOW_WS_ORIGIN`           | yes               | public hostname (`host[:port]`) allowed to open the websocket, i.e. the Spin ingress name                                                           |
| `NUM_PROCS`                       | no (default 2)    | Panel worker processes; sessions in one process share its caches (and its HDF5 lock, see Threads)                                                   |
| `NUM_THREADS`                     | no (default 4)    | thread pool per worker that callbacks run on, so one session's slow read does not freeze the others; `0` disables it (see Threads)                  |
| `LEDS_PREWARM`                    | no (default on)   | scan cycles and build the newest channelmap before forking workers, so the first session of each is warm; delays the listening socket; `0` disables |
| `LEDS_CACHE_TTL`                  | no (default 3600) | seconds a metadata-derived entry (channelmaps, statuses, calibration pars) is reused before being re-read                                           |
| `LEDS_SCAN_TTL`                   | no (default 300)  | seconds a directory scan is reused; bounds how long a newly-written run stays invisible                                                             |
| `LEDS_MAX_CACHED_RUN_SPECTRA`     | no (default 4)    | whole runs of per-hit energies held **per worker**; the dominant memory line                                                                        |
| `LEDS_MAX_CACHED_CAL_PARS`        | no (default 32)   | calibration parts of `par_hit`/`par_pht` files held **per worker**; ~5 MB each                                                                      |
| `LEDS_MAX_CACHED_VALIDATION_RUNS` | no (default 64)   | binned per-run rate summaries (all strings) held **per worker**; ~1.3 MB each                                                                       |

Sessions in one worker share the read-only data they have in common —
channelmaps, detector statuses, directory scans, per-run reductions — so a
second user on the same cycle starts nearly instantly and does not duplicate
that memory. All the bounds above are **per worker process**, so multiply by
`NUM_PROCS` when sizing the Spin memory limit.

Because that sharing includes the parsed metadata, an updated metadata checkout
is picked up within `LEDS_CACHE_TTL` rather than immediately. Lower it if the
deployment updates metadata often; restarting the service always picks it up at
once.

### Threads

`NUM_THREADS` (passed by the image to `leds serve --num-threads`, like
`NUM_PROCS` to `--num-procs`) runs callbacks on a thread pool instead of the
worker's single event loop. Without it, one session's slow operation — a first
visit to the Validation tab, a raw read stalling on CFS — freezes the UI of
every other session in that worker until it finishes. With it those sessions
keep responding, and work that is not HDF5 (parsing a channelmap, evaluating
cuts, building Bokeh models) runs in parallel.

What it does not buy: h5py serialises every HDF5 call behind one process-wide
lock, so two sessions' raw reads still take turns within a worker. `NUM_PROCS`
is the lever for that; the two combine.

Per-session state — the viewer's caches and open HDF5 handles, the spectrum
accumulator, the tab bookkeeping — is guarded by one lock per session, so a
session's own callbacks never overlap even though they run on the pool. Open
HDF5 handles are per session and never shared across sessions or forked workers.
A playback tick that arrives while the previous frame is still rendering is
dropped rather than queued.

### Login (Spin secrets)

Without any of the variables below the server is open. In all authenticated
setups set a **fixed** `LEDS_COOKIE_SECRET` — with an ephemeral one, logins
break on every restart and across the `NUM_PROCS` workers.

**Option A — shared password / user list:** set `LEDS_BASIC_AUTH` to either a
single shared password or a path to a mounted JSON file of
`{"username": "password"}`.

**Option B — LDAP (takes precedence over `LEDS_BASIC_AUTH`):** users sign in
with their directory credentials. Fill the values in from the LEGEND LDAP
administrators:

| Variable                                        | Required                        | Meaning                                                                             |
| ----------------------------------------------- | ------------------------------- | ----------------------------------------------------------------------------------- |
| `LEDS_LDAP_SERVER`                              | yes (enables LDAP)              | e.g. `ldaps://ldap.legend.example:636`                                              |
| `LEDS_LDAP_USER_DN_TEMPLATE`                    | direct-bind mode                | e.g. `uid={username},ou=people,dc=legend,dc=example`                                |
| `LEDS_LDAP_BIND_DN` / `LEDS_LDAP_BIND_PASSWORD` | search-bind mode                | read-only service account (as a Spin secret)                                        |
| `LEDS_LDAP_SEARCH_BASE`                         | search-bind mode                | subtree searched for the user entry                                                 |
| `LEDS_LDAP_USER_FILTER`                         | no (default `(uid={username})`) | search filter template                                                              |
| `LEDS_LDAP_GROUP_DN`                            | no                              | if set, only members (`member`/`uniqueMember`/`memberUid`) of this group may log in |
| `LEDS_LDAP_STARTTLS`                            | no                              | `1` upgrades an `ldap://` connection via StartTLS                                   |
| `LEDS_LDAP_CA_FILE`                             | no                              | CA bundle path (mount as a secret); defaults to certifi's bundle                    |

Configure **either** `LEDS_LDAP_USER_DN_TEMPLATE` (direct bind) **or** the three
search-bind variables; if both are present, direct bind wins. Server
certificates are always verified. Bad credentials, unknown users and non-members
all get the same generic login error; infrastructure failures are reported as
"authentication service unavailable" with details in the container log only.

To try the LDAP flow locally against a throwaway directory:

```bash
docker run --rm -d -p 1389:1389 -e LDAP_ADMIN_PASSWORD=adminpw \
  -e LDAP_USERS=alice -e LDAP_PASSWORDS=alicepw bitnamilegacy/openldap

LEDS_LDAP_SERVER=ldap://localhost:1389 \
LEDS_LDAP_USER_DN_TEMPLATE="cn={username},ou=users,dc=example,dc=org" \
LEDS_COOKIE_SECRET=devsecret leds serve /path/to/cycle
```
