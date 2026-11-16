"""Reproduce pipeline init error with logging (debug session 3795b7)."""
import json
import inspect
import time
import traceback
from pathlib import Path

LOG = Path(__file__).resolve().parent / ".cursor" / "debug-3795b7.log"


def log(hypothesis_id: str, message: str, data: dict, run_id: str = "pre-fix") -> None:
    # #region agent log
    payload = {
        "sessionId": "3795b7",
        "hypothesisId": hypothesis_id,
        "location": "debug_boardgame.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
        "runId": run_id,
    }
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    # #endregion


def main() -> None:
    import json
    from sklearn.pipeline import Pipeline
    import pandas as pd
    import numpy as np

    nb_path = Path(__file__).resolve().parent / "kaggle.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
    for c in code_cells:
        c["source"] = "".join(c.get("source", []))
    model_src = ""
    for c in code_cells:
        if "class BoardGameFeatures" in c["source"] and "LGBMRegressor" in c["source"]:
            model_src = c["source"]
            break

    log("H2", "model cell found", {"found": bool(model_src), "len": len(model_src)})

    ns: dict = {"pd": pd, "np": np}
    prep = "\n".join(
        c["source"]
        for c in code_cells[:8]
        if "read_csv" in c["source"] or "drop(['ID'" in c["source"]
    )
    exec(prep, ns)
    try:
        exec(model_src.split("pipe = Pipeline")[0], ns)
        BGF = ns["BoardGameFeatures"]
        sig = str(inspect.signature(BGF.__init__))
        log("H1", "init signature after exec", {"signature": sig})
        inst = BGF(mech_max_features=80, dom_max_features=30, min_df=30)
        log("H1", "direct instantiate ok", {"mech_max": inst.mech_max_features})
        pipe = Pipeline([("feat", BGF(mech_max_features=80, dom_max_features=30, min_df=30))])
        log("H3", "Pipeline construct ok", {"steps": [s[0] for s in pipe.steps]})
    except Exception as e:
        log("H1", "error", {"type": type(e).__name__, "msg": str(e), "tb": traceback.format_exc()})
        raise

    # H4: stale __init__ without params (user-edited variant)
    try:

        class BoardGameFeaturesStale:
            def __init__(self):
                pass

        BoardGameFeaturesStale(mech_max_features=80)
        log("H4", "stale class would not error", {})
    except TypeError as e:
        log("H4", "stale init reproduces user error", {"msg": str(e)})


if __name__ == "__main__":
    main()