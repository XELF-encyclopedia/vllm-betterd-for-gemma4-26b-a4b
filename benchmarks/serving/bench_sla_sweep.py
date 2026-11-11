# -*- coding: utf-8 -*-
"""SLA 极限压测脚本：测量流式 TTFT 与单 Session Decode TPS。

SLA 约束：
1. TTFT <= 6.0 秒
2. 单 Session TPS >= 30.0 tok/s
"""
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = 'http://127.0.0.1:8000/v1/chat/completions'
MODEL = 'base'
MAX_NEW = 2048

BASE_DOMAINS = [
    ("Semiconductor_EUV", "High-NA EUV Lithography and Sub-2nm Semiconductor Fabrication"),
    ("Quantum_Error_Correction", "Fault-Tolerant Quantum Computing and Surface Code Scalability"),
    ("Fusion_Energy_ITER", "Tokamak Plasma Confinement and Tungsten Divertor Integrity in ITER"),
    ("James_Webb_Exoplanets", "Atmospheric Characterization of Earth-Sized Exoplanets via JWST Transmission Spectroscopy"),
    ("CRISPR_Therapeutics", "In Vivo Epigenetic Editing and Cas9-Mediated Hemoglobinopathy Therapies"),
    ("Global_Shipping_RedSea", "Geopolitical Chokepoint Vulnerabilities and Cape of Good Hope Logistics Re-routing"),
    ("Central_Bank_Liquidity", "Quantitative Tightening Dynamics and Sovereign Debt Yield Curve Inversions"),
    ("DeepSea_Mining_Ecology", "Benthic Ecosystem Vulnerabilities and Polymetallic Nodule Extraction in the CCZ"),
    ("Hypersonic_Aerodynamics", "Boundary Layer Transition and Scramjet Combustion Diagnostics at Mach 7"),
    ("SolidState_Batteries", "Solid Electrolyte Interphase Degradation and Lithium Dendrite Suppression"),
    ("AI_Protein_Folding", "De Novo Macromolecular Design via Diffusion-Based Structural Generative Models"),
    ("Arctic_Ice_Albedo", "Cryospheric Positive Feedback Loops and Transpolar Sea Route Geoeconomics"),
    ("Gravitational_Waves", "Pulsar Timing Array Discoveries and Stochastic Supermassive Black Hole Mergers"),
    ("Carbon_Capture_DAC", "Direct Air Capture Thermodynamic Efficiencies and Metal-Organic Framework Sorbents"),
    ("Neuromorphic_Hardware", "Event-Driven Spiking Neural Networks and Memristor Crossbar Arrays"),
    ("Neurotech_BCI", "Implantable High-Density Intracortical Microelectrode Arrays and Neural Prosthetics"),
    ("Nuclear_Propulsion", "Bimodal Nuclear Thermal and Electric Propulsion for Deep-Space Interplanetary Missions"),
    ("High_Entropy_Alloys", "Refractory High-Entropy Alloys for Ultra-High-Temperature Aerospace Applications"),
    ("Organoid_Medicine", "Patient-Derived Microfluidic Organ-on-Chip Platforms for Personalized Oncology"),
    ("Metamaterial_Shields", "Elastodynamic Phononic Crystals for Resonant Seismic Ground Motion Attenuation"),
    ("Perovskite_PV", "Tandem Silicon-Perovskite Photovoltaic Cells and Long-Term Thermal-Moisture Degradation"),
    ("Synthetic_Biology", "Minimal Genome Engineering and Xenobiotic Nucleic Acid Incorporation"),
    ("Laser_Communication", "Optical Deep Space Communications via Pulse-Position Modulation and Single-Photon Arrays"),
    ("Graphene_Supercaps", "Vertical Carbon Nanotube Intercalation for High-Rate Electrochemical Supercapacitors"),
    ("EGS_Geothermal", "Supercritical Enhanced Geothermal Systems and Hydraulic Stimulation Microseismicity"),
    ("Gene_Drives", "Homing Endonuclease Gene Drives for Invasive Pest Vector Eradication"),
    ("High_Frequency_Trading", "Sub-Microsecond FPGA Order Routing and Asymmetric Market Microstructure"),
    ("Exoplanet_Biosignatures", "Atmospheric Disequilibrium Detection of Methane and Phosphine on Rocky Worlds"),
    ("Marine_Biodiversity", "High-Seas Biodiversity Treaty Enforcement via Satellite Remote Sensing Telemetry"),
    ("Plant_Genomics", "CRISPR-Edited C4 Photosynthetic Pathways for Climate-Resilient Agricultural Crop Yields"),
    ("Subsea_Telecomm", "Trans-Oceanic Fiber-Optic Cable Physical Security and Subsea Power Feed Topologies"),
    ("Microbiome_Immuno", "Gut Microbiota-Derived Short-Chain Fatty Acids and Checkpoint Inhibitor Responsiveness"),
]

PARTS = [
    ("Part_I", "Fundamental Microscopic Conservation Laws and Mathematical Formulations"),
    ("Part_II", "Empirical Testbed Sensor Telemetry and Stress Threshold Diagnostics"),
    ("Part_III", "Upstream Precursor Purification and Industrial Gigafactory Logistics"),
    ("Part_IV", "Geopolitical Strategic Chokepoints and Sovereign Regulatory Governance")
]

TOPICS = []
for tid, title in BASE_DOMAINS:
    for pid, pdesc in PARTS:
        TOPICS.append((f"{tid}_{pid}", f"{title} - {pdesc}",
                       f"This technical investigation evaluates {title.lower()} focusing on {pdesc.lower()}."))

def generate_article(topic_id, title, seed_text):
    paragraphs = []
    paragraphs.append(f"# STRATEGIC TECHNICAL MONOGRAPH: {title.upper()}\n\n")
    paragraphs.append(f"## Executive Overview\n{seed_text} Global industrial consortia and defense laboratories are actively re-engineering baseline architectures to surpass legacy thermodynamic limits. This monograph presents empirical metrics, mathematical boundary conditions, and supply chain dependencies.\n\n")
    
    sections = [
        ("Thermodynamic and Micro-Structural Envelopes",
         "The operational envelope is dictated by state transitions across interfacial boundaries. Non-equilibrium thermodynamic models show that energy dissipation follows power-law decay profiles when compensated by active harmonic feedback loops."),
        ("Empirical Validation Telemetry and Stress Metrics",
         "Under continuous duty cycling at elevated thermal and mechanical limits, multi-channel diagnostic telemetry captured real-time strain tensors and signal degradation rates across thousands of cycles."),
        ("Industrial Scale-Up and Precursor Supply Chains",
         "Commercial deployment hinges on the yield stability of high-purity upstream precursor chemical reagents. Bilateral strategic agreements and domestic manufacturing incentives aim to mitigate critical dependencies on concentrated foreign suppliers."),
        ("Future Research Roadmaps and Closed-Loop Optimization",
         "Next-generation systems will couple embedded quantum sensors with low-latency edge inference processors to implement predictive fault mitigation prior to physical threshold violation.")
    ]
    
    for sec_idx, (sec_title, sec_body) in enumerate(sections):
        paragraphs.append(f"### Section {sec_idx + 1}: {sec_title}\n")
        expanded = []
        for rep in range(2):
            expanded.append(f"{sec_body} Diagnostic trials in test regime {rep+1} demonstrated consistent repeatability with statistical variance constrained within 3.2 percent across all operational nodes.")
        paragraphs.append(" ".join(expanded) + "\n\n")
        
    paragraphs.append("## Concluding Synthesis\n")
    paragraphs.append("Achieving technological sovereignty requires systemic co-design spanning material crystallography, high-throughput manufacturing, and closed-loop algorithmic autonomy.\n")
    
    return "".join(paragraphs)

def build_articles(count):
    articles = []
    for idx, (topic_id, title, seed) in enumerate(TOPICS[:count]):
        art_text = generate_article(topic_id, title, seed)
        articles.append({
            "id": idx,
            "topic": topic_id,
            "title": title,
            "prompt": (
                f"你是顶尖的科技与地缘战略翻译与技术分析专家。请仔细阅读以下英文专业长文：\n\n"
                f"'''\n{art_text}\n'''\n\n"
                f"请完成以下出版级任务：\n"
                f"1. 将全文准确、地道、优雅地翻译成高质量简体中文，保持所有专业学术词汇与工程术语的严谨性。\n"
                f"2. 针对本文涉及的核心技术突破，撰写一份结构化技术影响与未来地缘战略推演报告。\n"
            )
        })
    return articles

def run_single_stream(item):
    """发起流式请求，精确记录 TTFT 和每 Token 解码耗时。"""
    payload = json.dumps({
        'model': MODEL,
        'messages': [{'role': 'user', 'content': item['prompt']}],
        'max_tokens': MAX_NEW,
        'temperature': 0.7,
        'top_p': 0.9,
        'stream': True,
        'stream_options': {'include_usage': True}
    }).encode('utf-8')
    
    req = urllib.request.Request(URL, data=payload, headers={'Content-Type': 'application/json'})
    t0 = time.time()
    t_first = None
    in_tokens = 0
    out_tokens = 0
    chunk_count = 0
    
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            for raw_line in resp:
                line = raw_line.decode('utf-8').strip()
                if not line or not line.startswith('data: '):
                    continue
                data_str = line[6:]
                if data_str == '[DONE]':
                    break
                try:
                    data = json.loads(data_str)
                except Exception:
                    continue
                
                # 检查首字
                choices = data.get('choices', [])
                if choices:
                    delta = choices[0].get('delta', {})
                    content = delta.get('content', '')
                    if content and t_first is None:
                        t_first = time.time()
                    if content:
                        chunk_count += 1
                        
                # 检查 usage
                if 'usage' in data and data['usage']:
                    u = data['usage']
                    in_tokens = u.get('prompt_tokens', 0)
                    out_tokens = u.get('completion_tokens', 0)

        t_end = time.time()
        if t_first is None:
            t_first = t_end
        ttft = t_first - t0
        decode_time = max(0.001, t_end - t_first)
        total_time = max(0.001, t_end - t0)
        
        # 若未拿到 usage，用 chunk 估计
        if out_tokens == 0:
            out_tokens = chunk_count
            
        decode_tps = out_tokens / decode_time
        e2e_tps = out_tokens / total_time
        
        return {
            'ok': True,
            'id': item['id'],
            'topic': item['topic'],
            'ttft': ttft,
            'decode_time': decode_time,
            'total_time': total_time,
            'in_tokens': in_tokens,
            'out_tokens': out_tokens,
            'decode_tps': decode_tps,
            'e2e_tps': e2e_tps,
        }
    except Exception as e:
        return {
            'ok': False,
            'id': item['id'],
            'topic': item['topic'],
            'err': f"{type(e).__name__}: {e}",
            'total_time': time.time() - t0,
        }

def evaluate_concurrency(c):
    print(f"\n{'='*75}")
    print(f"=== 开始评估并发规模: {c} 路并发 (SLA 标准: TTFT <= 6.0s, TPS >= 30.0 tok/s) ===")
    print(f"{'='*75}")
    articles = build_articles(c)
    t_start = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=c) as ex:
        futs = [ex.submit(run_single_stream, item) for item in articles]
        done_cnt = 0
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            done_cnt += 1
            if res['ok']:
                sla_ttft = "✅" if res['ttft'] <= 6.0 else "❌"
                sla_tps = "✅" if res['decode_tps'] >= 30.0 else "❌"
                print(f"[{done_cnt:3d}/{c}] [{res['topic']}] "
                      f"TTFT: {res['ttft']:.2f}s {sla_ttft} | "
                      f"Decode TPS: {res['decode_tps']:.1f} {sla_tps} | "
                      f"Out: {res['out_tokens']} tok | 耗时: {res['total_time']:.1f}s", flush=True)
            else:
                print(f"[{done_cnt:3d}/{c}] [{res['topic']}] ❌ 失败: {res.get('err')}", flush=True)

    wall_time = time.time() - t_start
    succ = [r for r in results if r['ok']]
    
    if not succ:
        print(f"❌ 全部请求失败!")
        return {'concurrency': c, 'succ_rate': 0, 'pass_sla': False}

    tot_in = sum(r['in_tokens'] for r in succ)
    tot_out = sum(r['out_tokens'] for r in succ)
    
    ttfts = sorted(r['ttft'] for r in succ)
    dec_tps = sorted(r['decode_tps'] for r in succ)
    
    p50_ttft = ttfts[len(ttfts)//2]
    p95_ttft = ttfts[int(len(ttfts)*0.95)]
    max_ttft = ttfts[-1]
    
    min_tps = dec_tps[0]
    p05_tps = dec_tps[int(len(dec_tps)*0.05)]
    p50_tps = dec_tps[len(dec_tps)//2]
    p95_tps = dec_tps[int(len(dec_tps)*0.95)]
    
    ttft_pass = sum(1 for t in ttfts if t <= 6.0)
    tps_pass = sum(1 for s in dec_tps if s >= 30.0)
    sla_full_pass = sum(1 for r in succ if r['ttft'] <= 6.0 and r['decode_tps'] >= 30.0)
    
    print("\n" + "-"*75)
    print(f"【{c} 并发指标总结】")
    print(f"成功率: {len(succ)}/{c} ({len(succ)/c*100:.1f}%) | 总耗时: {wall_time:.1f}s")
    print(f"集群有效输出吞吐: {tot_out/wall_time:.1f} tokens/s (总生成 {tot_out} tokens)")
    print(f"--- TTFT (首字延迟) ---")
    print(f"  Min: {ttfts[0]:.2f}s | P50: {p50_ttft:.2f}s | P95: {p95_ttft:.2f}s | Max: {max_ttft:.2f}s")
    print(f"  TTFT <= 6.0s 达标率: {ttft_pass}/{len(succ)} ({ttft_pass/len(succ)*100:.1f}%)")
    print(f"--- 单 Session Decode TPS (生成速率) ---")
    print(f"  Min: {min_tps:.1f} | P05: {p05_tps:.1f} | P50: {p50_tps:.1f} | P95: {p95_tps:.1f} tok/s")
    print(f"  TPS >= 30.0 达标率: {tps_pass}/{len(succ)} ({tps_pass/len(succ)*100:.1f}%)")
    print(f"--- 综合 SLA (TTFT <= 6s 且 TPS >= 30) ---")
    print(f"  双达标率: {sla_full_pass}/{len(succ)} ({sla_full_pass/len(succ)*100:.1f}%)")
    print("-"*75 + "\n")
    
    return {
        'concurrency': c,
        'succ_rate': len(succ) / c,
        'wall_time': wall_time,
        'agg_out_tps': tot_out / wall_time,
        'p50_ttft': p50_ttft,
        'p95_ttft': p95_ttft,
        'max_ttft': max_ttft,
        'ttft_pass_rate': ttft_pass / len(succ),
        'min_tps': min_tps,
        'p05_tps': p05_tps,
        'p50_tps': p50_tps,
        'tps_pass_rate': tps_pass / len(succ),
        'sla_full_pass_rate': sla_full_pass / len(succ),
    }

if __name__ == '__main__':
    target_c = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    evaluate_concurrency(target_c)
