"""Client for the device backend's ``/worker/*`` API (``vbb_contracts/machine_labels.md`` §3).

Every response is the device backend's envelope ``{code, message, action, data}`` with
``code == 200`` on success (HTTP status is 200 either way). Transport failures, HTTP 5xx and the
envelope's service-unavailable / maintenance codes are retried with backoff; anything else is the
backend saying no, and is raised at once as ``BackendError``.

Retrying is safe because the contract makes every call idempotent: register with the same
``run_id`` and arguments returns the existing run, finish / activate / revert to the status the run
already has return it, push is idempotent on ``(sighting_id, run_id)``. A call whose answer was lost
may therefore be replayed (the backend's maintenance mode even rewrites a handled request's code to
5003 after the fact, ``ApiResponse.php:54-58``).

``export_from_backend`` turns one facility-day's pulls into the same ``Export`` the offline CSV
loader builds, so production runs the code Phase A measured. The token never appears in logs,
errors or files: it is checked to be one printable word when the client is made, redirects are
refused (urllib would carry the header to the new host), and every error text is redacted.
"""

from __future__ import annotations

import ipaddress
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import replace
from datetime import date, datetime, timedelta
from typing import Callable, Iterator
from zoneinfo import ZoneInfo

from .data import BOARD_STATES, Export, Person, Sighting, local_day

OK = 200
RETRY_CODES = {500, 5003}  # ERROR_CODE_SERVICE_UNAVAILABLE, ERROR_CODE_MAINTAINED_MODE
PAGE = 500

# (url, body bytes, headers, timeout) -> (http status, response body bytes)
Transport = Callable[[str, bytes, dict, float], tuple[int, bytes]]


class BackendError(RuntimeError):
    def __init__(self, path: str, code, message: str):
        super().__init__(f"{path}: backend code {code}: {message}")
        self.path, self.code, self.message = path, code, message


TOKEN_RE = re.compile(r"^[\x21-\x7e]{16,512}$")  # one printable ASCII word: no spaces, newlines, controls


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, f"redirect refused (to {urllib.parse.urlsplit(newurl).netloc})",
                                     headers, fp)


_OPENER = urllib.request.build_opener(_NoRedirect)


def urllib_transport(url: str, body: bytes, headers: dict, timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:  # 4xx/5xx with a body, or a refused redirect
        if 300 <= e.code < 400:
            raise BackendError(urllib.parse.urlsplit(url).path, f"HTTP {e.code}", "redirect refused") from None
        return e.code, e.read() or b""


def check_base_url(url: str) -> str:
    """https only, except to this machine (tests, an SSH tunnel)."""
    u = urllib.parse.urlsplit(url)
    host = (u.hostname or "").lower()
    loopback = host == "localhost"
    try:
        loopback = loopback or ipaddress.ip_address(host).is_loopback
    except ValueError:
        pass
    if u.scheme == "https" or (u.scheme == "http" and loopback):
        return url.rstrip("/")
    raise ValueError(f"backend URL must be https:// (http only to localhost): {u.scheme}://{u.netloc}")


class BackendClient:
    def __init__(self, base_url: str, token: str, *, timeout_s: float = 30.0, retries: int = 5,
                 backoff_s: float = 2.0, transport: Transport | None = None,
                 sleep: Callable[[float], None] = time.sleep, user_agent: str = "vbb-reid-server"):
        if not TOKEN_RE.match(token or ""):
            raise ValueError("the worker token must be one printable word of 16-512 characters "
                             "(check the token file for spaces or extra lines)")
        self.base = check_base_url(base_url)
        self._token = token
        self.timeout_s, self.retries, self.backoff_s = timeout_s, retries, backoff_s
        self.transport = transport or urllib_transport
        self.sleep = sleep
        self.user_agent = user_agent

    def __repr__(self) -> str:  # never the token
        return f"BackendClient({self.base!r})"

    def redact(self, text: str) -> str:
        return str(text).replace(self._token, "***")

    # ------------------------------------------------------------- calls --
    def call(self, path: str, payload: dict):
        url = f"{self.base}/{path.lstrip('/')}"
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   "Authorization": f"Bearer {self._token}", "User-Agent": self.user_agent}
        last = None
        for attempt in range(self.retries + 1):
            try:
                status, raw = self.transport(url, body, headers, self.timeout_s)
            except BackendError:
                raise
            except (OSError, TimeoutError) as e:  # URLError is an OSError
                last = self.redact(f"transport error: {e.__class__.__name__}: {e}")
            except Exception as e:  # e.g. http.client refusing a header: never retried, never echoed raw
                raise BackendError(path, "client", self.redact(f"{e.__class__.__name__}: {e}")) from None
            else:
                if status >= 500:
                    last = f"HTTP {status}"
                else:
                    try:
                        env = json.loads(raw.decode("utf-8") or "null")
                    except ValueError:
                        raise BackendError(path, f"HTTP {status}", "response is not JSON") from None
                    if not isinstance(env, dict) or "code" not in env:
                        raise BackendError(path, f"HTTP {status}", "response is not the {code, message, data} envelope")
                    code = env.get("code")
                    if code == OK:
                        return env.get("data")
                    if code not in RETRY_CODES:
                        raise BackendError(path, code, self.redact(str(env.get("message") or "")))
                    last = self.redact(f"backend code {code}: {env.get('message') or ''}")
            if attempt < self.retries:
                self.sleep(self.backoff_s * (2 ** attempt))
        raise BackendError(path, "retries exhausted", last or "")

    def pages(self, path: str, payload: dict) -> Iterator[dict]:
        cursor = None
        seen: set[str] = set()
        while True:
            data = self.call(path, {**payload, "cursor": cursor, "limit": PAGE})
            if not isinstance(data, dict):
                raise BackendError(path, "bad data", "page is not an object")
            yield data
            cursor = data.get("next_cursor")
            if not cursor:
                return
            if cursor in seen:
                raise BackendError(path, "bad cursor", f"next_cursor {cursor!r} repeats; paging would never end")
            seen.add(cursor)

    # --------------------------------------------------------- endpoints --
    def register(self, run_id: str, facility_id: int, day: str, model_id: str, worker_id: str,
                 thresholds: dict) -> dict:
        return self.call("worker/runs/register", {"run_id": run_id, "facility_id": facility_id, "day": day,
                                                  "model_id": model_id, "worker_id": worker_id,
                                                  "thresholds": thresholds})

    def pull_sightings(self, facility_id: int, day: str) -> tuple[list[dict], str | None]:
        rows: list[dict] = []
        tz = None
        for page in self.pages("worker/sightings/pull", {"facility_id": facility_id, "day": day}):
            rows.extend(page.get("sightings") or [])
            tz = tz or page.get("time_zone")
        return rows, tz


    def pull_references(self, facility_id: int, since_ts: int) -> tuple[list[dict], list[dict], str | None]:
        refs: list[dict] = []
        roster: list[dict] | None = None
        tz = None
        for page in self.pages("worker/references/pull", {"facility_id": facility_id, "since_ts": since_ts}):
            refs.extend(page.get("references") or [])
            if roster is None and page.get("roster") is not None:
                roster = page["roster"]
            tz = tz or page.get("time_zone")
        return refs, roster or [], tz

    def push_labels(self, run_id: str, rows: list[dict]) -> dict:
        if not 1 <= len(rows) <= PAGE:
            raise ValueError(f"a push carries 1..{PAGE} rows, got {len(rows)}")
        return self.call("worker/labels/push", {"run_id": run_id, "rows": rows})

    def finish(self, run_id: str, status: str, counts: dict[str, int], error: str | None = None) -> dict:
        return self.call("worker/runs/finish", {"run_id": run_id, "status": status, "counts": counts,
                                                "error": error})



# ------------------------------------------------------------------ export --
def day_start_ms(day: str, tz_name: str) -> int:
    d = date.fromisoformat(day)
    return int(datetime(d.year, d.month, d.day, tzinfo=ZoneInfo(tz_name)).timestamp() * 1000)


def _s(v) -> str | None:
    return None if v is None or v == "" else str(v)


def export_from_backend(client: BackendClient, facility_id: int, day: str, *, window_days: int,
                        tz_name: str = "Asia/Shanghai") -> Export:
    """One facility-day as an ``Export``: the day's sightings (probes) and the references.

    - Probes: board prediction as reported; ``human_labeled`` rows keep their overlay kind
      (``human_state``), so they are decided but never pushed.
    - References: human ``known`` labels (already scoped by the backend to this facility's roster,
      person ids folded), on sightings captured since ``day - window_days``; a same-day label lands
      on its probe row.
    - Roster: this facility's registered ids.
    """
    fac = str(facility_id)
    rows, tz_s = client.pull_sightings(facility_id, day)
    tz = tz_name
    since = day_start_ms((date.fromisoformat(day) - timedelta(days=window_days)).isoformat(), tz)
    refs, roster, tz_r = client.pull_references(facility_id, since)
    told = {t for t in (tz_s, tz_r) if t}
    if not told:
        raise BackendError("worker/*/pull", "time zone", f"the backend sent no time_zone for facility {fac}")
    if told != {tz_name}:
        raise BackendError("worker/*/pull", "time zone",
                           f"facility {fac} is in {', '.join(sorted(told))}, the run assumed {tz_name}: "
                           f"pass --tz {sorted(told)[0]}")

    sightings: dict[str, Sighting] = {}
    for r in rows:
        st = r.get("board_state") or "uncertain"
        st = st if st in BOARD_STATES else "uncertain"
        hs = None
        if r.get("human_labeled"):  # any overlay (known, stranger, reset): decided, never pushed (run.py)
            hs = r.get("human_state") or "stranger"
            hs = hs if hs in ("known", "stranger", "reset") else "stranger"
        sightings[str(r["sighting_id"])] = Sighting(
            sighting_id=str(r["sighting_id"]), did=str(r.get("did") or ""), room_id=str(r.get("room_id") or ""),
            facility_id=fac, track_id=r.get("track_id"), board_state=st,
            board_person_id=_s(r.get("board_person_id")) if st == "known" else None,
            provisional_person_id=_s(r.get("provisional_person_id")), best_match_score=None,
            crop_quality=r.get("crop_quality"), crop_is_face=bool(r.get("crop_is_face")), crop_count=1,
            image_url=_s(r.get("image_url")), capture_ts=int(r["capture_ts"]),
            track_started_ts=r.get("track_started_ts"), track_ended_ts=r.get("track_ended_ts"),
            human_state=hs, human_person_id=None, corrected_ts=None, day=day)

    for r in refs:
        sid = str(r["sighting_id"])
        rfac = str(r.get("facility_id") or fac)
        cap = int(r["capture_ts"])
        if sid in sightings and rfac == fac:
            sightings[sid] = replace(sightings[sid], human_state="known", human_person_id=str(r["person_id"]),
                                     corrected_ts=int(r["corrected_ts"]))
            continue
        sightings[sid] = Sighting(
            sighting_id=sid, did=str(r.get("did") or ""), room_id="", facility_id=rfac, track_id=None,
            board_state="uncertain", board_person_id=None, provisional_person_id=None, best_match_score=None,
            crop_quality=None, crop_is_face=bool(r.get("crop_is_face")), crop_count=1,
            image_url=_s(r.get("image_url")), capture_ts=cap, track_started_ts=None, track_ended_ts=None,
            human_state="known", human_person_id=str(r["person_id"]), corrected_ts=int(r["corrected_ts"]),
            day=str(r.get("day") or local_day(cap, tz)))

    people = {str(p["person_id"]): Person(person_id=str(p["person_id"]), facility_id=fac,
                                          person_type=_s(p.get("person_type")), role=_s(p.get("role")),
                                          tentative=bool(p.get("tentative")), merged_into=_s(p.get("merged_into")))
              for p in roster}
    facs = {fac} | {s.facility_id for s in sightings.values()}
    ordered = sorted(sightings.values(), key=lambda s: (s.facility_id, s.capture_ts, s.sighting_id))
    return Export(root=None, sightings=ordered, people=people,
                  facility_tz={f: tz for f in facs}, facility_org={f: "brand" for f in facs})
