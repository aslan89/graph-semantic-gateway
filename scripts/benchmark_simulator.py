import argparse
import asyncio
import json
import logging
from pathlib import Path
import httpx
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("benchmark.simulator")


async def send_single_query(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    base_url: str,
    mode: str,
    row: dict,
) -> dict:
    async with semaphore:
        headers = {
            "Content-Type": "application/json",
            "X-Gateway-Mode": mode,
            # Deney izolasyonu: Test sorgularının önbelleğe yazılıp sonraki modları etkilemesini önler
            "X-Skip-Cache-Write": "true",
        }
        payload = {
            "messages": [{"role": "user", "content": row["question2"]}],
        }

        resp = await client.post(f"{base_url.rstrip('/')}/v1/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        body = resp.json()
        tel = body["telemetry"]

        is_dup = int(row["is_duplicate"])
        cache_status = tel["cache_status"]
        is_hit = cache_status == "HIT"

        # Sınıflandırma sonucu (Passthrough modu hariç)
        if mode == "passthrough":
            outcome = "BYPASS"
        elif is_dup == 1 and is_hit:
            outcome = "TP"  # Doğru anlamsal eşleşme yakalandı
        elif is_dup == 1 and not is_hit:
            outcome = "FN"  # Eşdeğer soru kaçırıldı
        elif is_dup == 0 and not is_hit:
            outcome = "TN"  # Tuzak soru başarıyla reddedildi
        else:
            outcome = "FP"  # HATA: Farklı soruya yanlış önbellek yanıtı verildi (Cache Poisoning/Error)

        return {
            "mode": mode,
            "pair_id": row["id"],
            "question1": row["question1"],
            "question2": row["question2"],
            "is_duplicate": is_dup,
            "cache_status": cache_status,
            "outcome": outcome,
            "total_latency_ms": tel["total_latency_ms"],
            "feature_extraction_ms": tel["feature_extraction_ms"],
            "cache_lookup_ms": tel["cache_lookup_ms"],
            "llm_upstream_ms": tel["llm_upstream_ms"],
            "similarity_score": tel["similarity_score"],
            "applied_threshold": tel["applied_threshold"],
            "prompt_tokens": tel["prompt_tokens"],
            "completion_tokens": tel["completion_tokens"],
            "estimated_cost_saved": tel["estimated_cost_saved"],
        }


def summarize_mode_metrics(df_mode: pd.DataFrame, mode: str) -> dict:
    latencies = df_mode["total_latency_ms"].to_numpy()
    summary = {
        "mode": mode,
        "total_requests": int(len(df_mode)),
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_latency_ms": round(float(np.percentile(latencies, 99)), 2),
        "avg_extraction_ms": round(float(df_mode["feature_extraction_ms"].mean()), 2),
        "avg_lookup_ms": round(float(df_mode["cache_lookup_ms"].mean()), 2),
        "total_prompt_tokens_billed": int(df_mode["prompt_tokens"].sum()),
        "total_completion_tokens_billed": int(df_mode["completion_tokens"].sum()),
    }

    if mode == "passthrough":
        summary.update(
            {
                "TP": 0,
                "TN": 0,
                "FP": 0,
                "FN": 0,
                "cache_hit_rate_pct": 0.0,
                "accuracy_pct": None,
                "precision_pct": None,
                "recall_tpr_pct": None,
                "specificity_tnr_pct": None,
                "false_positive_rate_pct": None,
                "f1_score": None,
            }
        )
        return summary

    tp = int((df_mode["outcome"] == "TP").sum())
    tn = int((df_mode["outcome"] == "TN").sum())
    fp = int((df_mode["outcome"] == "FP").sum())
    fn = int((df_mode["outcome"] == "FN").sum())

    total = tp + tn + fp + fn
    accuracy = ((tp + tn) / total * 100) if total > 0 else 0.0
    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
    specificity = (tn / (tn + fp) * 100) if (tn + fp) > 0 else 0.0
    fpr = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall) / 100) if (precision + recall) > 0 else 0.0
    hit_rate = ((tp + fp) / total * 100) if total > 0 else 0.0

    summary.update(
        {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "cache_hit_rate_pct": round(hit_rate, 2),
            "accuracy_pct": round(accuracy, 2),
            "precision_pct": round(precision, 2),
            "recall_tpr_pct": round(recall, 2),
            "specificity_tnr_pct": round(specificity, 2),
            "false_positive_rate_pct": round(fpr, 2),
            "f1_score": round(f1, 4),
        }
    )
    return summary


async def run_benchmark(
    base_url: str,
    eval_path: Path,
    modes: list[str],
    concurrency: int,
    out_csv: Path,
    out_json: Path,
) -> None:
    if not eval_path.exists():
        raise FileNotFoundError(
            f"'{eval_path}' bulunamadı. Önce 'python -m scripts.etl_qqp' betiğini çalıştırın."
        )

    eval_df = pd.read_csv(eval_path)
    records = eval_df.to_dict(orient="records")
    all_results: list[dict] = []
    summaries: list[dict] = []

    limits = httpx.Limits(max_keepalive_connections=concurrency * 2, max_connections=concurrency * 4)
    timeout = httpx.Timeout(60.0)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        # Sunucu sağlık kontrolü
        health = await client.get(f"{base_url.rstrip('/')}/health")
        health.raise_for_status()

        for mode in modes:
            logger.info("=== Deney Başlatılıyor | Mod: %s | İstek Sayısı: %d | Eşzamanlılık: %d ===", mode, len(records), concurrency)
            semaphore = asyncio.Semaphore(concurrency)
            tasks = [
                send_single_query(client, semaphore, base_url, mode, row)
                for row in records
            ]

            mode_results = []
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Simülasyon [{mode}]"):
                res = await coro
                mode_results.append(res)

            all_results.extend(mode_results)
            df_mode = pd.DataFrame(mode_results)
            summaries.append(summarize_mode_metrics(df_mode, mode))

    # Detaylı ve özet çıktıları kaydet
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_results).to_csv(out_csv, index=False)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    # Konsola tez tablosu formatında özet bas
    summary_df = pd.DataFrame(summaries)
    print("\n" + "=" * 95)
    print("TEZ DENEYSEL KIYASLAMA ÖZETİ (HYBRID vs VECTOR_ONLY vs PASSTHROUGH)")
    print("=" * 95)
    print(summary_df.to_string(index=False))
    print("=" * 95)
    print(f"Detaylı istek logları : {out_csv}")
    print(f"Özet JSON raporu      : {out_json}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Gateway Anlamsal Önbellek Kıyaslama Simülatörü")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000", help="Gateway sunucu adresi")
    parser.add_argument("--eval-path", type=Path, default=Path("data/eval_pairs.csv"), help="Değerlendirme CSV yolu")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["hybrid", "vector_only", "passthrough"],
        help="Karşılaştırılacak modlar",
    )
    parser.add_argument("--concurrency", type=int, default=5, help="Eşzamanlı istemci sayısı")
    parser.add_argument("--out-csv", type=Path, default=Path("data/benchmark_results.csv"))
    parser.add_argument("--out-json", type=Path, default=Path("data/benchmark_summary.json"))
    args = parser.parse_args()

    asyncio.run(
        run_benchmark(
            base_url=args.base_url,
            eval_path=args.eval_path,
            modes=args.modes,
            concurrency=args.concurrency,
            out_csv=args.out_csv,
            out_json=args.out_json,
        )
    )


if __name__ == "__main__":
    main()