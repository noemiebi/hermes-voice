"""tests/out/stt-real/{A,B,C,D}.json -> 분석(summary.json, summary.md, results.json) + extras/stt-compare.html + extras/clips-review/*.wav.

  .venv/bin/python tests/stt_real_report.py

정답이 없으므로 모델 간 일치만 본다. 비교는 모델 원문(raw)을 정규화(공백·문장부호 제거, 소문자)해서 한다.
"""
from __future__ import annotations

import html
import json
import os
import re
import shutil
import statistics
import sys
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from server.stt import _HALLU_PATTERNS, _norm, is_hallucination  # noqa: E402

OUT = ROOT / "tests" / "out" / "stt-real"
CLIPS = ROOT / "tests" / "clips"
WEB = ROOT / "extras"   # 실험 페이지: 앱(web/)에서 서빙하지 않음
REVIEW = WEB / "clips-review"
LABELS = {"A": "A 현재 turbo", "B": "B turbo-ko", "C": "C Qwen3-ASR", "D": "D large-v3"}
CORE = ["A", "B", "C"]
HALLU_CPS = 15.0  # 정규화 글자/초(짧은 클립은 1초로 봄)가 이보다 많으면 환각 의심
# 고유명사 목록(환각 판정에서 제외). 사람 이름 같은 개인 어휘는 여기 말고 PROPER_EXTRA 환경변수(쉼표 구분)로 넣는다.
PROPER = ["헤르메스", "hermes", "카카오톡", "카톡", "클로드", "claude", "오퍼스", "opus", "소네트",
          "sonnet", "텔레그램", "telegram", "네이버", "지메일", "gmail",
          "코덱스", "codex", "제미나이", "gemini", "gpt", "구글", "google", "유튜브", "아이폰", "iphone", "깃허브", "github"]
PROPER += [w.strip() for w in os.environ.get("PROPER_EXTRA", "").split(",") if w.strip()]


_NUM = re.compile(r"(?:\d+|(?:열|스물|서른|마흔|쉰|한|두|세|네|다섯|여섯|일곱|여덟|아홉|일|이|삼|사|오|육|칠|팔|구|십|백|천|만)+)"
                  r"(?=개|시|도|명|통|번|분|살|월|주|달|원|층|장|권|건)")


def loose(n: str) -> str:
    """정규화 문자열에서 숫자 표기(3개/세 개, 12도/열두 도)를 #로 통일."""
    return _NUM.sub("#", n)


def sim(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def suspect(raw: str, secs: float) -> str:
    if not raw.strip():
        return "empty"
    if is_hallucination(raw):
        return "hallu-pattern"
    if any(p.search(raw) for p in _HALLU_PATTERNS[:-1] + [re.compile(r"광고를\s*포함")]):
        return "hallu-phrase"  # 길이와 무관하게 자막·광고 문구가 섞임
    if len(_norm(raw)) > HALLU_CPS * max(secs, 1.0):
        return "too-dense"
    return ""


def diff_html(base: str, other: str) -> str:
    """other를 base와 어절 단위로 비교해 다른 부분을 <mark>."""
    bw, ow = base.split(), other.split()
    sm = SequenceMatcher(None, [_norm(w) for w in bw], [_norm(w) for w in ow])
    out = []
    for op, _, _, j1, j2 in sm.get_opcodes():
        seg = html.escape(" ".join(ow[j1:j2]))
        if seg:
            out.append(seg if op == "equal" else f"<mark>{seg}</mark>")
    return " ".join(out)


def main() -> None:
    runs = {k: json.loads((OUT / f"{k}.json").read_text(encoding="utf-8")) for k in "ABCD" if (OUT / f"{k}.json").exists()}
    keys = list(runs)
    core = [k for k in CORE if k in runs]
    names = sorted(runs["A"]["clips"])
    rows, results = [], {}
    for n in names:
        secs = runs["A"]["clips"][n]["seconds"]
        r = {"seconds": secs, "orig": {}}
        txt = (CLIPS / f"{n}.txt")
        if txt.exists():
            r["orig"] = dict(l.split("\t", 1) for l in txt.read_text(encoding="utf-8").splitlines() if "\t" in l)
        for k in keys:
            c = runs[k]["clips"].get(n, {})
            raw = c.get("raw", "")
            r[k] = {"raw": raw, "text": c.get("text", ""), "reason": c.get("reason", ""), "ms": c.get("ms"),
                    "runs": c.get("runs"), "norm": _norm(raw), "loose": loose(_norm(raw)), "suspect": suspect(raw, secs)}
        allt = " ".join(r[k]["raw"] for k in keys).lower()
        r["proper"] = sorted({p for p in PROPER if p in allt})
        r["mismatch_strict"] = len({r[k]["norm"] for k in core}) > 1
        r["mismatch"] = len({r[k]["loose"] for k in core}) > 1  # 숫자 표기 차이는 무시
        results[n] = r
        rows.append((n, r))

    # 요약
    summary = {"clips": len(names), "models": {}, "pairs": {}, "hallu_cps": HALLU_CPS}
    for k in keys:
        cl = runs[k]["clips"]
        summary["models"][k] = {
            "model": runs[k]["model"], "load_ms": runs[k]["load_ms"],
            "ms_median": statistics.median(c["ms"] for c in cl.values()),
            "ms_p90": sorted(c["ms"] for c in cl.values())[int(0.9 * len(cl))],
            "rtf_median": round(statistics.median(c["ms"] / 1000 / max(c["seconds"], 0.1) for c in cl.values()), 3),
            "empty": sum(1 for r in results.values() if r[k]["suspect"] == "empty"),
            "suspect": sum(1 for r in results.values() if r[k]["suspect"]),
            "filtered": sum(1 for r in results.values() if r[k]["reason"]),
        }
    for a, b in combinations(keys, 2):
        ex = sum(1 for r in results.values() if r[a]["norm"] == r[b]["norm"])
        ms = statistics.mean(sim(r[a]["norm"], r[b]["norm"]) for r in results.values())
        lx = sum(1 for r in results.values() if r[a]["loose"] == r[b]["loose"])
        summary["pairs"][f"{a}-{b}"] = {"exact": round(ex / len(names), 3), "exact_n": ex, "char_sim": round(ms, 3),
                                        "exact_loose": round(lx / len(names), 3)}
    if all(k in runs for k in CORE):
        L = "loose"
        summary["loose_all_three_differ"] = sum(1 for r in results.values()
                                                if len({r["A"][L], r["B"][L], r["C"][L]}) == 3)
        summary["loose_only_A_differs"] = sum(1 for r in results.values() if r["B"][L] == r["C"][L] != r["A"][L])
        summary["loose_only_B_differs"] = sum(1 for r in results.values() if r["A"][L] == r["C"][L] != r["B"][L])
        summary["loose_only_C_differs"] = sum(1 for r in results.values() if r["A"][L] == r["B"][L] != r["C"][L])
        summary["all_three_differ"] = sum(1 for r in results.values()
                                          if len({r["A"]["norm"], r["B"]["norm"], r["C"]["norm"]}) == 3)
        summary["only_A_differs"] = sum(1 for r in results.values()
                                        if r["B"]["norm"] == r["C"]["norm"] != r["A"]["norm"])
        summary["only_B_differs"] = sum(1 for r in results.values()
                                        if r["A"]["norm"] == r["C"]["norm"] != r["B"]["norm"])
        summary["only_C_differs"] = sum(1 for r in results.values()
                                        if r["A"]["norm"] == r["B"]["norm"] != r["C"]["norm"])
        summary["all_same"] = sum(1 for r in results.values() if not r["mismatch"])
    summary["mismatch_strict"] = sum(1 for r in results.values() if r["mismatch_strict"])
    summary["mismatch"] = sum(1 for r in results.values() if r["mismatch"])
    summary["proper"] = sum(1 for r in results.values() if r["proper"])
    summary["proper_mismatch"] = sum(1 for r in results.values() if r["proper"] and r["mismatch"])

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps({"summary": summary, "clips": results}, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    md = ["| 모델 | 지연 중앙값(ms) | p90 | RTF | 로드(ms) | A와 완전일치 | 숫자표기 무시 | A와 문자유사도 | 빈 결과 | 환각 의심(빈 포함) | 필터로 버림 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in keys:
        m = summary["models"][k]
        p = summary["pairs"].get(f"A-{k}", {"exact": 1.0, "char_sim": 1.0, "exact_loose": 1.0})
        md.append(f"| {k} {m['model']} | {m['ms_median']:.0f} | {m['ms_p90']} | {m['rtf_median']} | {m['load_ms']} | "
                  f"{p['exact']:.0%} | {p['exact_loose']:.0%} | {p['char_sim']:.3f} | {m['empty']} | {m['suspect']} | {m['filtered']} |")
    md += ["", "쌍별: " + ", ".join(f"{p} 일치 {v['exact']:.0%}(숫자무시 {v['exact_loose']:.0%})/유사 {v['char_sim']:.3f}" for p, v in summary["pairs"].items()),
           "", json.dumps({k: v for k, v in summary.items() if k not in ("models", "pairs")}, ensure_ascii=False)]
    (OUT / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    # 클립 복사
    REVIEW.mkdir(parents=True, exist_ok=True)
    for n in names:
        dst = REVIEW / f"{n}.wav"
        if not dst.exists():
            shutil.copy2(CLIPS / f"{n}.wav", dst)
    gi = ROOT / ".gitignore"
    if "extras/clips-review/" not in gi.read_text(encoding="utf-8").splitlines():
        with gi.open("a", encoding="utf-8") as f:
            f.write("extras/clips-review/\n")

    # 페이지
    e = html.escape
    head_cells = []
    for k in keys:
        m = summary["models"][k]
        p = summary["pairs"].get(f"A-{k}")
        agree = f" · A와 일치 {p['exact']:.0%}(숫자 표기 무시 {p['exact_loose']:.0%})" if p else ""
        head_cells.append(f"<li><b>{e(LABELS[k])}</b> <span>{e(m['model'])}</span><br>지연 중앙값 {m['ms_median']:.0f} ms"
                          f" · 로드 {m['load_ms']} ms{agree} · 환각 의심 {m['suspect']}</li>")
    stat = (f"클립 {summary['clips']}개 · A/B/C 불일치 {summary['mismatch']}(숫자 표기까지 따지면 {summary['mismatch_strict']})"
            f" · 셋 다 다름 {summary.get('loose_all_three_differ', '-')} · A만 다름 {summary.get('loose_only_A_differs', '-')}"
            f" · B만 {summary.get('loose_only_B_differs', '-')} · C만 {summary.get('loose_only_C_differs', '-')}"
            f" · 고유명사 포함 {summary['proper']}")
    body = []
    for n, r in rows:
        lines = []
        for k in keys:
            raw = r[k]["raw"]
            shown = e(raw) if k == "A" else diff_html(r["A"]["raw"], raw)
            note = []
            if r[k]["suspect"]:
                note.append(r[k]["suspect"])
            if r[k]["reason"]:
                note.append(f"버림:{r[k]['reason']}")
            elif r[k]["text"] and r[k]["text"] != raw:
                note.append("교정→ " + r[k]["text"])
            nt = f'<em>{e(" · ".join(note))}</em>' if note else ""
            lines.append(f'<div class="t"><span class="k k{k}">{k}</span><div>{shown or "<i>(빈 결과)</i>"}{nt}</div></div>')
        if r["mismatch_strict"] and not r["mismatch"]:
            pn_extra = '<span class="tag2">숫자 표기만 다름</span>'
        else:
            pn_extra = ""
        cls = "row" + (" mis" if r["mismatch"] else "") + (" pn" if r["proper"] else "")
        pn = f'<span class="tag">{e(", ".join(r["proper"]))}</span>' if r["proper"] else ""
        body.append(f'<section class="{cls}"><div class="h"><b>{e(n)}</b><span>{r["seconds"]:.1f}초</span>{pn}{pn_extra}</div>'
                    f'<audio controls preload="none" src="clips-review/{e(n)}.wav"></audio>{"".join(lines)}</section>')
    page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><title>STT 모델 비교</title>
<style>:root{{color-scheme:dark}}body{{margin:0;padding:20px 16px 48px;background:#0f1115;color:#e6e8ec;font:16px/1.5 -apple-system,system-ui,sans-serif}}h1{{font-size:20px;margin:0 0 4px}}p{{color:#9aa1ad;margin:0 0 12px;font-size:14px}}ul{{list-style:none;padding:0;margin:0 0 12px;font-size:13px;color:#9aa1ad}}ul li{{margin:4px 0}}ul b{{color:#e6e8ec}}ul span{{font-size:12px}}.f{{display:flex;gap:8px;margin:0 0 16px;position:sticky;top:0;background:#0f1115;padding:8px 0;z-index:1}}.f button{{background:#171a21;color:#e6e8ec;border:1px solid #2a2f3a;border-radius:999px;padding:6px 14px;font:inherit;font-size:14px}}.f button.on{{background:#3b6fd8;border-color:#3b6fd8}}.f small{{align-self:center;color:#9aa1ad}}section{{background:#171a21;border-radius:14px;padding:12px 14px;margin-bottom:10px}}section.mis{{background:#1f1a24}}.h{{display:flex;gap:10px;align-items:baseline;font-size:13px;color:#9aa1ad;margin-bottom:6px;flex-wrap:wrap}}.h b{{color:#e6e8ec;font-weight:600}}.tag{{color:#e0b85a}}.tag2{{color:#7f8796}}audio{{width:100%;height:36px;margin-bottom:6px}}.t{{display:flex;gap:10px;margin:4px 0;font-size:15px}}.k{{flex:none;width:20px;height:20px;border-radius:6px;font-size:12px;line-height:20px;text-align:center;font-weight:700;margin-top:2px}}.kA{{background:#3a4150}}.kB{{background:#2f5d4a}}.kC{{background:#5a3f6e}}.kD{{background:#5c4a2a}}mark{{background:#5b4a1c;color:#fff;border-radius:3px;padding:0 2px}}em{{display:block;color:#9aa1ad;font-size:12px;font-style:normal}}i{{color:#9aa1ad}}.hide{{display:none}}</style></head>
<body><h1>STT 모델 비교 (실사용 클립)</h1><p>{e(stat)}<br>각 줄은 모델 원문. B/C/D의 <mark>표시</mark>는 A와 다른 어절. 불일치 = 공백·문장부호 제거, 숫자 표기(3개/세 개) 통일 후 A·B·C가 완전히 같지 않음. 빈 결과·환각 의심은 각 줄 아래 회색 글씨.</p>
<ul>{"".join(head_cells)}</ul>
<div class="f"><button data-f="all">전부</button><button data-f="mis" class="on">불일치만</button><button data-f="pn">고유명사</button><small id="n"></small></div>
<main>{"".join(body)}</main>
<script>
const rows=[...document.querySelectorAll('section.row')],btns=[...document.querySelectorAll('.f button')],cnt=document.getElementById('n');
function show(f){{let n=0;for(const r of rows){{const ok=f==='all'||(f==='mis'&&r.classList.contains('mis'))||(f==='pn'&&r.classList.contains('pn'));r.classList.toggle('hide',!ok);if(ok)n++;}}btns.forEach(b=>b.classList.toggle('on',b.dataset.f===f));cnt.textContent=n+'개';}}
btns.forEach(b=>b.onclick=()=>show(b.dataset.f));
document.addEventListener('play',ev=>{{for(const a of document.querySelectorAll('audio'))if(a!==ev.target)a.pause();}},true);
show('mis');
</script></body></html>
"""
    (WEB / "stt-compare.html").write_text(page, encoding="utf-8")
    print("page:", WEB / "stt-compare.html", "rows", len(rows))


if __name__ == "__main__":
    main()
