# -*- coding: utf-8 -*-
"""平滑到达压测：128 路并发，以业务真实间隔（间隔 0.04 秒）连续注入，测量平稳期 TTFT 与单 Session TPS。"""
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, '/root')
from bench_sla import build_articles, run_single_stream

CONCURRENCY = 128
articles = build_articles(CONCURRENCY)

print(f"\n{'='*75}")
print(f"=== 真实平滑业务流压测: {CONCURRENCY} 路并发 (错峰间隔 0.04s 注入) ===")
print(f"=== SLA 目标: TTFT <= 6.0s, 单流 TPS >= 30.0 tok/s ===")
print(f"{'='*75}")

t_start = time.time()
results = []

def send_with_delay(idx_item):
    idx, item = idx_item
    time.sleep(idx * 0.04)  # 128 个请求在 5.1 秒内均匀注入
    return run_single_stream(item)

with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
    futs = [ex.submit(send_with_delay, (i, item)) for i, item in enumerate(articles)]
    done_cnt = 0
    for fut in as_completed(futs):
        res = fut.result()
        results.append(res)
        done_cnt += 1
        if res['ok']:
            sla_ttft = "✅" if res['ttft'] <= 6.0 else "❌"
            sla_tps = "✅" if res['decode_tps'] >= 30.0 else "❌"
            print(f"[{done_cnt:3d}/{CONCURRENCY}] [{res['topic']}] "
                  f"TTFT: {res['ttft']:.2f}s {sla_ttft} | "
                  f"Decode TPS: {res['decode_tps']:.1f} {sla_tps} | "
                  f"Out: {res['out_tokens']} tok | 耗时: {res['total_time']:.1f}s", flush=True)

wall_time = time.time() - t_start
succ = [r for r in results if r['ok']]
tot_out = sum(r['out_tokens'] for r in succ)
ttfts = sorted(r['ttft'] for r in succ)
dec_tps = sorted(r['decode_tps'] for r in succ)

ttft_pass = sum(1 for t in ttfts if t <= 6.0)
tps_pass = sum(1 for s in dec_tps if s >= 30.0)
sla_pass = sum(1 for r in succ if r['ttft'] <= 6.0 and r['decode_tps'] >= 30.0)

print("\n" + "-"*75)
print(f"【{CONCURRENCY} 并发平滑业务流总结】")
print(f"成功率: {len(succ)}/{CONCURRENCY} ({len(succ)/CONCURRENCY*100:.1f}%) | 总耗时: {wall_time:.1f}s")
print(f"集群总输出吞吐: {tot_out/wall_time:.1f} tok/s (总生成 {tot_out} tokens)")
print(f"--- TTFT (首字延迟) ---")
print(f"  Min: {ttfts[0]:.2f}s | P50: {ttfts[len(ttfts)//2]:.2f}s | P95: {ttfts[int(len(ttfts)*0.95)]:.2f}s | Max: {ttfts[-1]:.2f}s")
print(f"  TTFT <= 6.0s 达标率: {ttft_pass}/{len(succ)} ({ttft_pass/len(succ)*100:.1f}%)")
print(f"--- 单 Session Decode TPS ---")
print(f"  Min: {dec_tps[0]:.1f} | P50: {dec_tps[len(dec_tps)//2]:.1f} | P95: {dec_tps[int(len(dec_tps)*0.95)]:.1f} tok/s")
print(f"  TPS >= 30.0 达标率: {tps_pass}/{len(succ)} ({tps_pass/len(succ)*100:.1f}%)")
print(f"--- 双达标率 ---: {sla_pass}/{len(succ)} ({sla_pass/len(succ)*100:.1f}%)")
print("-"*75)
