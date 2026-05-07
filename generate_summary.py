import json
import os

sizes = [1, 2, 5]
results = {}

for size in sizes:
    file = f"seva_results_4060_{size}k.json"
    if os.path.exists(file):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            results[size] = data
    else:
        results[size] = None

with open("final_summary.md", "w") as f:
    f.write("# SEVA Benchmark Cross-Corpus Results\n\n")
    f.write("> [!NOTE]\n")
    f.write("> These results were generated dynamically using the `benchmark.py` testing suite across 1k, 2k, and 5k corpus sizes.\n\n")
    f.write("| Corpus Size | ASR (%) | Doc FPR (%) | Query FPR (%) | Mean Latency (ms) | P95 Latency (ms) |\n")
    f.write("| ----------- | ------- | ----------- | ------------- | ----------------- | ---------------- |\n")
    for size in sizes:
        if results[size] is not None:
            data = results[size]
            det = data.get("detection", {})
            lat = data.get("latency_ms", {})
            
            asr = det.get("asr_pct", 0.0)
            doc_fpr = det.get("doc_fpr_pct", 0.0)
            query_fpr = det.get("query_fpr_pct", 0.0)
            mean_lat = lat.get("pipeline_mean", 0.0)
            p95_lat = lat.get("pipeline_p95", 0.0)
            
            # Additional detail to show targets
            f.write(f"| {size}k | {asr:.2f}% | {doc_fpr:.2f}% | {query_fpr:.2f}% | {mean_lat:.2f} | {p95_lat:.2f} |\n")
        else:
            f.write(f"| {size}k | Error | Error | Error | Error | Error |\n")

print("Created final_summary.md correctly")
