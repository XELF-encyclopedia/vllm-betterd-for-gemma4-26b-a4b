# -*- coding: utf-8 -*-
"""128 路完全并发深度文章压测套件：128 篇完全不重样的学术/科技长文。"""
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = 'http://127.0.0.1:8000/v1/chat/completions'
MODEL = 'base'
MAX_NEW = 2048
CONCURRENCY = 128

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

print(f"生成的独立话题数量: {len(TOPICS)}")

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

ARTICLES = []
for idx, (topic_id, title, seed) in enumerate(TOPICS[:CONCURRENCY]):
    art_text = generate_article(topic_id, title, seed)
    ARTICLES.append({
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

def make_req(item):
    return json.dumps({
        'model': MODEL,
        'messages': [{'role': 'user', 'content': item['prompt']}],
        'max_tokens': MAX_NEW,
        'temperature': 0.7,
        'top_p': 0.9,
    }).encode('utf-8')

def send_one(item):
    payload = make_req(item)
    req = urllib.request.Request(URL, data=payload, headers={'Content-Type': 'application/json'})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            d = json.loads(r.read())
        u = d['usage']
        dt = time.time() - t0
        out_tokens = u['completion_tokens']
        in_tokens = u['prompt_tokens']
        return {
            'ok': True, 'id': item['id'], 'topic': item['topic'], 'dt': dt,
            'in_tokens': in_tokens, 'out_tokens': out_tokens,
            'tps': out_tokens / dt if dt > 0 else 0
        }
    except Exception as e:
        return {'ok': False, 'id': item['id'], 'topic': item['topic'], 'dt': time.time() - t0, 'err': f"{type(e).__name__}: {e}"}

print(f"=== 开始发起 {CONCURRENCY} 路并发不重样长文压测 (max_tokens={MAX_NEW}) ===")
t_start = time.time()
results = []

with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
    futs = [ex.submit(send_one, item) for item in ARTICLES]
    done_cnt = 0
    for fut in as_completed(futs):
        res = fut.result()
        results.append(res)
        done_cnt += 1
        if res['ok']:
            print(f"[{done_cnt:3d}/{CONCURRENCY}] [{res['topic']}] 成功: {res['in_tokens']} in, {res['out_tokens']} out, 耗时 {res['dt']:.1f}s ({res['tps']:.1f} tok/s)", flush=True)
        else:
            print(f"[{done_cnt:3d}/{CONCURRENCY}] [{res['topic']}] 失败: 耗时 {res['dt']:.1f}s, 错误: {res.get('err')}", flush=True)

total_time = time.time() - t_start
succ = [r for r in results if r['ok']]
tot_in = sum(r['in_tokens'] for r in succ)
tot_out = sum(r['out_tokens'] for r in succ)

print("\n" + "="*70)
print(f"=== {CONCURRENCY} 篇完全不重样长文压测总结 (并发={CONCURRENCY}, max_tokens={MAX_NEW}) ===")
print(f"总耗时: {total_time:.1f} 秒, 成功率: {len(succ)}/{CONCURRENCY} ({len(succ)/CONCURRENCY*100:.1f}%)")
if succ:
    print(f"总处理输入 Tokens: {tot_in} (平均每篇 {tot_in/len(succ):.0f} tokens)")
    print(f"总生成输出 Tokens: {tot_out} (平均每篇 {tot_out/len(succ):.0f} tokens)")
    print(f"有效端到端吞吐: 输入 {tot_in/total_time:.1f} tok/s, 输出 {tot_out/total_time:.1f} tok/s")
    lats = sorted(r['dt'] for r in succ)
    print(f"请求耗时分布: Min={lats[0]:.1f}s, P50={lats[len(lats)//2]:.1f}s, P95={lats[int(len(lats)*0.95)]:.1f}s, Max={lats[-1]:.1f}s")
print("="*70)
