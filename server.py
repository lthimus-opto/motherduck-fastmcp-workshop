"""Your MCP server for San Francisco 311 data.

FastMCP Gateway runs this file. Its entrypoint setting is `server.py:mcp`.
"""

import logging
import re
from collections import Counter, defaultdict
from statistics import median

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from sf311.data import BY_ID, LOCAL_NAMES, NEIGHBORHOODS, REQUESTS, SERVICES, closest
from sf311.kit import caller_email

mcp = FastMCP("sf-311")
logger = logging.getLogger(__name__)

HOUSE_NUMBER = re.compile(r"^(\d+)\s+([^,]+)")

GROUP_BY_FIELDS = {
    "neighborhood": lambda r: r.neighborhood,
    "service": lambda r: r.service,
    "agency": lambda r: r.agency,
    "source": lambda r: r.source,
    "weekday": lambda r: r.opened.strftime("%A"),
    "status": lambda r: r.status,
}
MEDIAN_GROUP_BY_FIELDS = {"neighborhood": GROUP_BY_FIELDS["neighborhood"], "agency": GROUP_BY_FIELDS["agency"]}
MIN_CLOSED_FOR_MEDIAN = 10
MAX_ROWS = 50


def _check_service(service: str | None) -> None:
    if service is not None and service not in SERVICES:
        suggestions = closest(service, SERVICES)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ToolError(f"Unknown service {service!r}.{hint}")


def _check_neighborhood(neighborhood: str | None) -> None:
    if neighborhood is not None and neighborhood not in NEIGHBORHOODS:
        suggestions = closest(neighborhood, NEIGHBORHOODS)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ToolError(f"Unknown neighborhood {neighborhood!r}.{hint}")


def _mask_address(address: str) -> str:
    """Turn "994 HARRISON ST, SAN FRANCISCO, ..." into "900 block of HARRISON ST"."""
    match = HOUSE_NUMBER.match(address)
    if not match:
        return address
    block = int(match.group(1)) // 100 * 100
    return f"{block} block of {match.group(2).strip()}"


def _masked_dict(request) -> dict:
    row = request.as_dict()
    row["address"] = _mask_address(row["address"])
    return row


def _matching(service: str | None = None, neighborhood: str | None = None, status: str | None = None) -> list:
    _check_service(service)
    _check_neighborhood(neighborhood)
    return [
        r
        for r in REQUESTS
        if (service is None or r.service == service)
        and (neighborhood is None or r.neighborhood == neighborhood)
        and (status is None or r.status == status)
    ]


@mcp.tool(annotations={"readOnlyHint": True})
def dataset_info() -> dict:
    """Describe the 311 dataset this server answers questions about.

    Call this first to learn the date range and the exact service and
    neighborhood names that other tools accept.
    """
    return {
        "description": "San Francisco 311 service requests filed in one week.",
        "first_opened": min(request.opened for request in REQUESTS).isoformat(),
        "last_opened": max(request.opened for request in REQUESTS).isoformat(),
        "requests": len(REQUESTS),
        "services": SERVICES,
        "neighborhoods": NEIGHBORHOODS,
        "local_neighborhood_names": len(LOCAL_NAMES),
    }


@mcp.tool(annotations={"readOnlyHint": True})
def search_requests(
    service: str | None = None,
    neighborhood: str | None = None,
    status: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Find individual 311 requests, newest first.

    Use this to look at specific cases, such as examples for a neighborhood
    or service. Use `count_requests` instead when you only need a total, and
    `median_close_hours` instead for how long requests take to close.

    `status` is "Open" or "Closed". `limit` is 1 to 50 (default 20).
    Returns at most `limit` requests, each with the fields of a 311 request
    (request_id, opened, closed, status, service, subtype, agency,
    neighborhood, local_name, address, source). `address` is masked to the
    block, such as "900 block of HARRISON ST".
    """
    if not 1 <= limit <= MAX_ROWS:
        raise ToolError(f"limit must be between 1 and {MAX_ROWS}, not {limit}.")
    if status is not None and status not in ("Open", "Closed"):
        raise ToolError(f"status must be 'Open' or 'Closed', not {status!r}.")
    rows = _matching(service, neighborhood, status)
    rows.sort(key=lambda r: r.opened, reverse=True)
    return [_masked_dict(r) for r in rows[:limit]]


@mcp.tool(annotations={"readOnlyHint": True})
def count_requests(
    service: str | None = None,
    neighborhood: str | None = None,
    group_by: str | None = None,
) -> int | dict[str, int]:
    """Count 311 requests, optionally broken down by group.

    Use this for totals and breakdowns. Use `search_requests` instead when
    you need to see the individual requests, and `median_close_hours`
    instead for how long requests take to close.

    `group_by` is one of "neighborhood", "service", "agency", "source",
    "weekday", "status". Without it, returns a single count. With it,
    returns a dict of group name to count, largest group first, with at
    most 50 groups.
    """
    if group_by is not None and group_by not in GROUP_BY_FIELDS:
        raise ToolError(f"group_by must be one of {sorted(GROUP_BY_FIELDS)}, not {group_by!r}.")
    rows = _matching(service, neighborhood)
    if group_by is None:
        return len(rows)
    key = GROUP_BY_FIELDS[group_by]
    counts = Counter(key(r) for r in rows if key(r))
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return dict(ranked[:MAX_ROWS])


@mcp.tool(annotations={"readOnlyHint": True})
def median_close_hours(
    service: str | None = None,
    group_by: str | None = None,
) -> float | None | dict[str, float]:
    """Return the median hours from opened to closed for closed 311 requests.

    Use this for how fast requests get resolved. Use `count_requests`
    instead for totals, and `search_requests` instead to see individual
    requests.

    `group_by` is "neighborhood" or "agency". Without it, returns one
    number rounded to 0.1 hours, or None if no matching requests are closed.
    With it, returns a dict of group name to median hours, fastest first,
    leaving out groups with fewer than 10 closed requests, at most 50 groups.
    """
    if group_by is not None and group_by not in MEDIAN_GROUP_BY_FIELDS:
        raise ToolError(f"group_by must be one of {sorted(MEDIAN_GROUP_BY_FIELDS)}, not {group_by!r}.")
    rows = [r for r in _matching(service) if r.hours_to_close is not None]
    if group_by is None:
        return round(median(r.hours_to_close for r in rows), 1) if rows else None
    key = MEDIAN_GROUP_BY_FIELDS[group_by]
    hours = defaultdict(list)
    for r in rows:
        if key(r):
            hours[key(r)].append(r.hours_to_close)
    medians = {
        group: round(median(values), 1) for group, values in hours.items() if len(values) >= MIN_CLOSED_FOR_MEDIAN
    }
    return dict(sorted(medians.items(), key=lambda item: (item[1], item[0]))[:MAX_ROWS])


@mcp.tool(annotations={"readOnlyHint": True})
def count_open_requests(service: str | None = None, neighborhood: str | None = None) -> int:
    """Count 311 requests that are still open (not yet closed).

    Use this for "how many are open" questions. Use `count_requests`
    instead for a total regardless of status, or its `group_by="status"`
    for a full open/closed breakdown.
    """
    return len(_matching(service, neighborhood, status="Open"))


@mcp.tool(annotations={"readOnlyHint": True})
def get_request(request_id: str) -> dict:
    """Return one 311 request by its request_id.

    Use this when you already have a request_id, for example from
    `search_requests`. Use `search_requests` instead to find requests by
    service, neighborhood, or status. `address` is masked to the block,
    such as "900 block of HARRISON ST".
    """
    request = BY_ID.get(request_id)
    if request is None:
        raise ToolError(f"No request with id {request_id!r}. Use search_requests to find a valid request_id.")
    logger.info("get_request %s caller=%s", request_id, caller_email() or "local")
    return _masked_dict(request)


if __name__ == "__main__":
    mcp.run()
