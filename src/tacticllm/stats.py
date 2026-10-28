"""Statystyki taktyczne liczone z pozycji zawodników (+ opcjonalnie adnotacji punktów).

Wszystkie pozycje są przeliczane na "perspektywę drużyny":
    depth  = odległość od siatki w metrach (0 = przy siatce, 10 = linia końcowa)
    x_own  = 0 (ściana po stronie revésu) .. 10 (ściana po stronie drive'u)
dzięki temu obie pary i obie strony kortu da się porównywać bezpośrednio.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from .court import COURT_WIDTH, NET_Y
from .models import MatchInfo, PlayerObs, PointAnnotation, Segment
from .teams import team_of_side

NET_ZONE_M = 4.5  # obaj bliżej siatki niż to = para "trzyma siatkę"
BACK_ZONE_M = 7.0  # obaj dalej niż to = para w obronie
MAX_SPEED_MS = 9.0  # szybsze "przeskoki" to błędy detekcji, nie bieg
HEAT_BINS = (5, 5)  # siatka mapy cieplnej (x_own, depth) na połowie kortu


def own_coords(obs: PlayerObs) -> tuple[float, float]:
    if obs.side == "near":
        return obs.x, obs.y - NET_Y
    return COURT_WIDTH - obs.x, NET_Y - obs.y


def formation(depths: list[float]) -> str:
    if len(depths) < 2:
        return "unknown"
    if max(depths) < NET_ZONE_M:
        return "net"
    if min(depths) > BACK_ZONE_M:
        return "back"
    if min(depths) < NET_ZONE_M and max(depths) > BACK_ZONE_M:
        return "split"  # jeden przy siatce, drugi z tyłu
    return "transition"


def point_features(seg: Segment) -> dict:
    """Cechy jednego punktu dla obu drużyn."""
    frames: dict[tuple[str, float], list[PlayerObs]] = defaultdict(list)
    for o in seg.observations:
        frames[(team_of_side(seg, o.side), o.t)].append(o)

    result = {}
    for team in ("A", "B"):
        times = sorted(t for (tm, t) in frames if tm == team)
        states = [formation([own_coords(o)[1] for o in frames[(team, t)]]) for t in times]
        known = [s for s in states if s != "unknown"]

        # zmiany ustawienia siatka <-> tył (pomijamy stany przejściowe)
        gains = losses = 0
        prev = None
        for s in known:
            if s not in ("net", "back"):
                continue
            if prev == "back" and s == "net":
                gains += 1
            elif prev == "net" and s == "back":
                losses += 1
            prev = s

        players = {}
        for lane in ("drive", "reves"):
            track = sorted((o for t in times for o in frames[(team, t)] if o.lane == lane), key=lambda o: o.t)
            dist = 0.0
            for a, b in zip(track, track[1:]):
                dt = b.t - a.t
                step = float(np.hypot(b.x - a.x, b.y - a.y))
                if 0 < dt <= 1.0 and step / dt <= MAX_SPEED_MS:
                    dist += step
            players[lane] = {
                "distance_m": round(dist, 1),
                "avg_depth_m": round(mean(own_coords(o)[1] for o in track), 2) if track else None,
                "samples": len(track),
            }

        heat = np.zeros(HEAT_BINS)
        for t in times:
            for o in frames[(team, t)]:
                x_own, depth = own_coords(o)
                ix = min(int(np.clip(x_own, 0, COURT_WIDTH - 1e-6) / COURT_WIDTH * HEAT_BINS[0]), HEAT_BINS[0] - 1)
                iy = min(int(np.clip(depth, 0, 10 - 1e-6) / 10 * HEAT_BINS[1]), HEAT_BINS[1] - 1)
                heat[ix, iy] += 1

        result[team] = {
            "net_share": round(known.count("net") / len(known), 3) if known else None,
            "back_share": round(known.count("back") / len(known), 3) if known else None,
            "split_share": round(known.count("split") / len(known), 3) if known else None,
            "net_gains": gains,
            "net_losses": losses,
            "end_formation": known[-1] if known else "unknown",
            "players": players,
            "heat": heat.tolist(),
        }
    return result


def set_of(seg: Segment, ann: PointAnnotation | None, info: MatchInfo, board_set: int | None = None) -> int:
    """Kolejność źródeł: ręcznie podane początki setów, tablica wyników, opis modelu."""
    if info.set_starts_s:
        starts = sorted(info.set_starts_s)
        return max([i + 1 for i, s in enumerate(starts) if seg.start_s >= s], default=1)
    if board_set:
        return board_set
    if ann and ann.set_number:
        return ann.set_number
    return 1


def build_points(segments: list[Segment], annotations: dict[int, PointAnnotation], info: MatchInfo,
                 scores: dict[int, dict] | None = None) -> list[dict]:
    """scores: odczyty tablicy wyników (models.load_scores). Zwycięzca z tablicy jest pewniejszy niż
    ocena modelu wizyjnego, więc ma pierwszeństwo. Gdy opis modelu wskazuje innego zwycięzcę, jego
    zakończenie, ostatnie uderzenie i przebieg dotyczą złej pary – nie wchodzą do statystyk."""
    scores = scores or {}
    points = []
    for seg in segments:
        ann = annotations.get(seg.index)
        board = scores.get(seg.index, {})
        if ann is not None and not ann.is_rally and not board.get("winner"):
            continue
        feats = point_features(seg)
        ann_winner = team_of_side(seg, ann.winner_side) if ann and ann.winner_side != "unknown" else None
        winner = board.get("winner") or ann_winner
        conflict = bool(board.get("winner") and ann_winner and ann_winner != board["winner"])
        described = ann if ann and not conflict else None
        points.append({
            "index": seg.index,
            "start_s": seg.start_s,
            "duration_s": round(seg.duration, 1),
            "set": set_of(seg, ann, info, board.get("set")),
            "near_team": seg.near_team,
            "winner": winner,
            "winner_source": "scoreboard" if board.get("winner") else ("video_model" if ann_winner else None),
            "ending": described.ending if described else None,
            "last_shot": described.last_shot if described else None,
            "shots": ann.shots if ann else None,
            "lobs": ({team_of_side(seg, "near"): ann.lobs_near, team_of_side(seg, "far"): ann.lobs_far}
                     if ann else None),
            "server": team_of_side(seg, ann.server_side) if ann and ann.server_side != "unknown" else None,
            "tactics": ({team_of_side(seg, "near"): list(ann.tactics_near), team_of_side(seg, "far"): list(ann.tactics_far)}
                        if ann else None),
            "decisive_moment": described.decisive_moment if described else None,
            "notes": described.notes if described else None,
            "teams": feats,
        })
    return points


def _rate(won: int, total: int) -> float | None:
    return round(won / total, 3) if total else None


def aggregate(points: list[dict]) -> dict:
    """Podsumowanie dla każdej drużyny z listy punktów."""
    out = {}
    for team in ("A", "B"):
        opp = "B" if team == "A" else "A"
        feats = [p["teams"][team] for p in points]
        decided = [p for p in points if p["winner"]]
        won = [p for p in decided if p["winner"] == team]

        def avg(key):
            vals = [f[key] for f in feats if f[key] is not None]
            return round(mean(vals), 3) if vals else None

        by_end = {}
        for form in ("net", "back", "split", "transition"):
            subset = [p for p in decided if p["teams"][team]["end_formation"] == form]
            by_end[form] = {"points": len(subset), "win_rate": _rate(sum(p["winner"] == team for p in subset), len(subset))}

        by_length = {}
        for name, lo, hi in (("krótkie <6s", 0, 6), ("średnie 6-12s", 6, 12), ("długie >12s", 12, 1e9)):
            subset = [p for p in decided if lo <= p["duration_s"] < hi]
            by_length[name] = {"points": len(subset), "win_rate": _rate(sum(p["winner"] == team for p in subset), len(subset))}

        on_serve = [p for p in decided if p["server"] == team]
        on_return = [p for p in decided if p["server"] == opp]

        endings = defaultdict(int)
        for p in decided:
            if p["ending"] == "winner" and p["winner"] == team:
                endings["winners"] += 1
            elif p["ending"] == "unforced_error" and p["winner"] == opp:
                endings["unforced_errors"] += 1
            elif p["ending"] == "forced_error" and p["winner"] == team:
                endings["forced_errors_drawn"] += 1
        finishing = defaultdict(int)
        for p in won:
            if p["last_shot"]:
                finishing[p["last_shot"]] += 1

        lobs = [p["lobs"][team] for p in points if p["lobs"] and p["lobs"].get(team) is not None]

        # jak często para stosowała daną taktykę i ile punktów z nią wygrała
        with_tactics = [p for p in points if p["tactics"] is not None]
        usage = defaultdict(lambda: {"points": 0, "won": 0, "decided": 0})
        for p in with_tactics:
            for tag in set(p["tactics"].get(team, [])):
                usage[tag]["points"] += 1
                if p["winner"]:
                    usage[tag]["decided"] += 1
                    usage[tag]["won"] += p["winner"] == team
        tactics = {
            tag: {"points": u["points"], "share_of_points": _rate(u["points"], len(with_tactics)),
                  "win_rate": _rate(u["won"], u["decided"])}
            for tag, u in sorted(usage.items(), key=lambda kv: -kv[1]["points"])
        }

        players = {}
        for lane in ("drive", "reves"):
            dists = [f["players"][lane]["distance_m"] for f in feats if f["players"][lane]["samples"]]
            depths = [f["players"][lane]["avg_depth_m"] for f in feats if f["players"][lane]["avg_depth_m"] is not None]
            players[lane] = {
                "avg_distance_per_point_m": round(mean(dists), 1) if dists else None,
                "avg_depth_m": round(mean(depths), 2) if depths else None,
            }

        heat = np.sum([np.array(f["heat"]) for f in feats], axis=0) if feats else np.zeros(HEAT_BINS)
        heat_norm = (heat / heat.sum()).round(3).tolist() if heat.sum() else heat.tolist()

        out[team] = {
            "points_played": len(points),
            "points_with_result": len(decided),
            "points_won": len(won),
            "win_rate": _rate(len(won), len(decided)),
            "serve_win_rate": _rate(sum(p["winner"] == team for p in on_serve), len(on_serve)),
            "return_win_rate": _rate(sum(p["winner"] == team for p in on_return), len(on_return)),
            "avg_net_share": avg("net_share"),
            "avg_back_share": avg("back_share"),
            "avg_split_share": avg("split_share"),
            "net_gains_per_point": avg("net_gains"),
            "net_losses_per_point": avg("net_losses"),
            "win_rate_by_end_formation": by_end,
            "win_rate_by_rally_length": by_length,
            "endings": dict(endings),
            "winning_shots": dict(finishing),
            "lobs_per_point": round(mean(lobs), 2) if lobs else None,
            "tactics": tactics,
            "players": players,
            # wiersze: x_own (revés -> drive), kolumny: odległość od siatki (siatka -> tył)
            "heatmap": heat_norm,
        }
    return out


def compute_stats(segments: list[Segment], annotations: dict[int, PointAnnotation], info: MatchInfo,
                  scores: dict[int, dict] | None = None) -> dict:
    points = build_points(segments, annotations, info, scores)
    sets = sorted({p["set"] for p in points})
    return {
        "teams": {tid: info.team_name(tid) for tid in ("A", "B")},
        "has_annotations": bool(annotations),
        "winners_from_scoreboard": sum(p["winner_source"] == "scoreboard" for p in points),
        "points_count": len(points),
        "avg_point_duration_s": round(mean(p["duration_s"] for p in points), 1) if points else None,
        "overall": aggregate(points),
        "by_set": {str(s): aggregate([p for p in points if p["set"] == s]) for s in sets},
        "points": [{k: v for k, v in p.items() if k != "teams"} | {
            "net_share": {t: p["teams"][t]["net_share"] for t in ("A", "B")},
            "end_formation": {t: p["teams"][t]["end_formation"] for t in ("A", "B")},
        } for p in points],
    }


def save_heatmaps(stats: dict, out_dir: Path) -> list[Path]:
    """PNG z mapą cieplną pozycji każdej pary (perspektywa drużyny, siatka u góry)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for team, data in stats["overall"].items():
        heat = np.array(data["heatmap"]).T  # wiersze = głębokość
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(heat, cmap="magma", extent=(0, 10, 10, 0), aspect="equal")
        ax.axhline(NET_ZONE_M, color="white", ls="--", lw=0.8)
        ax.axhline(6.95, color="cyan", lw=0.8)
        ax.set_xlabel("revés  ←  szerokość [m]  →  drive")
        ax.set_ylabel("odległość od siatki [m]")
        ax.set_title(f"{stats['teams'][team]} – pozycje")
        fig.tight_layout()
        path = out_dir / f"heatmap_{team}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        files.append(path)
    return files
