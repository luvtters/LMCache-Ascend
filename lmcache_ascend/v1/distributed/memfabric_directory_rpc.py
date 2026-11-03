# SPDX-License-Identifier: Apache-2.0
"""JSON control client for one fixed-rank shared L2 adapter on a trusted network."""

import http.client
import json
from urllib.parse import quote, urlsplit


class DirectoryClient:
    """A rank may claim once per run. Never replay a request after loss of a reply."""

    def __init__(self, url, group, rank, contract):
        endpoint = urlsplit(url)
        if (endpoint.scheme not in ("http", "https") or not endpoint.hostname
                or endpoint.username is not None or endpoint.password is not None
                or endpoint.query or endpoint.fragment):
            raise ValueError("shared L2 requires the fixed-group HTTP(S) endpoint")
        self._connection = (http.client.HTTPSConnection if endpoint.scheme == "https"
                            else http.client.HTTPConnection)
        self._host, self._port = endpoint.hostname, endpoint.port
        self._path = endpoint.path.rstrip("/") + "/runs/" + quote(group, safe="") + "/directory"
        self._rank = rank
        reply = self._post("claim", dict(rank=rank, contract=contract))
        if not all(isinstance(reply.get(k), str) and reply[k] for k in ("epoch", "claim")):
            raise RuntimeError("invalid shared L2 directory claim reply")
        self._epoch, self._claim = reply["epoch"], reply["claim"]
        self._slot_size = contract["slot_size_mb"] << 20
        self.on_retired = None

    @property
    def epoch(self):
        return self._epoch

    def _post(self, method, message):
        body = json.dumps(message, allow_nan=False).encode("utf-8")
        if len(body) > 65536:
            raise ValueError("shared L2 metadata exceeds 65536 bytes")
        connection = self._connection(self._host, self._port, timeout=5)
        try:
            connection.request("POST", self._path + "/" + method, body,
                               {"Content-Type": "application/json"})
            response = connection.getresponse()
            try:
                size = int(response.getheader("Content-Length", ""))
            except ValueError as exc:
                raise RuntimeError("missing shared L2 directory reply size") from exc
            if not 0 < size <= 65536 or response.getheader("Transfer-Encoding"):
                raise RuntimeError("invalid shared L2 directory reply size")
            data = response.read(size)
            if len(data) != size:
                raise RuntimeError("incomplete shared L2 directory reply")
            try:
                reply = json.loads(data)
            except (ValueError, RecursionError) as exc:
                raise RuntimeError("invalid shared L2 directory reply") from exc
            if not isinstance(reply, dict):
                raise RuntimeError("invalid shared L2 directory envelope")
            if (response.status == 422 and method == "call" and set(reply) == {"epoch", "error"}
                    and reply["epoch"] == self.epoch and isinstance(reply["error"], str)):
                raise ValueError(reply["error"])
            if response.status != 200:
                raise RuntimeError(f"shared L2 directory HTTP {response.status}: {reply.get('error')}")
            return reply
        except (OSError, http.client.HTTPException) as exc:
            raise RuntimeError("shared L2 directory connection failed; outcome unknown") from exc
        finally:
            connection.close()

    def call(self, op, **args):
        reply = self._post("call", dict(rank=self._rank, claim=self._claim, epoch=self.epoch,
                                       op=op, args=args))
        if set(reply) != {"epoch", "result", "retired"} or reply["epoch"] != self.epoch:
            raise RuntimeError("shared L2 directory reply has an invalid epoch or result")
        retired = reply.get("retired")
        if not isinstance(retired, list) or any(
                not isinstance(item, list) or len(item) != 2
                or not isinstance(item[0], str) or not item[0]
                or type(item[1]) is not int or not 0 < item[1] <= self._slot_size
                for item in retired):
            raise RuntimeError("invalid shared L2 retirement events")
        if retired and self.on_retired is not None:
            try:
                self.on_retired(retired)
            except Exception as exc:
                raise RuntimeError("failed to apply shared L2 retirement events") from exc
        return reply["result"]
