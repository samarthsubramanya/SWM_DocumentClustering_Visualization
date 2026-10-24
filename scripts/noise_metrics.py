import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari
from swm import db

for r in db.query("SELECT run_id FROM runs WHERE algo = 'hdbscan' ORDER BY run_id"):
    rows = db.query("""SELECT dc.cluster_id, dc.is_noise, d.label FROM doc_cluster dc
                       JOIN documents d ON d.id = dc.doc_id WHERE dc.run_id = %s""", (r["run_id"],))
    c = np.array([x["cluster_id"] for x in rows]); n = np.array([x["is_noise"] for x in rows])
    y = np.array([x["label"] for x in rows]); k = ~n
    print(f"{r['run_id']:32s} core-only NMI {nmi(y[k], c[k]):.3f} ARI {ari(y[k], c[k]):.3f} | "
          f"noise reassigned NMI {nmi(y, c):.3f} ARI {ari(y, c):.3f}")
    core = {x["doc_id"] for x in db.query(
        "SELECT doc_id FROM doc_cluster WHERE run_id = 'all-mpnet-base-v2__hdbscan' AND NOT is_noise")}
    for run in ("all-mpnet-base-v2__kmeans", "bge-base-en-v1.5__kmeans"):
        rows = [x for x in db.query("""SELECT dc.doc_id, dc.cluster_id, d.label FROM doc_cluster dc
            JOIN documents d ON d.id = dc.doc_id WHERE dc.run_id = %s""", (run,)) if x["doc_id"] in core]
        c = [x["cluster_id"] for x in rows]; y = [x["label"] for x in rows]
        print(f"{run:32s} on HDBSCAN core ({len(rows)} docs): NMI {nmi(y, c):.3f} ARI {ari(y, c):.3f}")