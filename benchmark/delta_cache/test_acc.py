import json
import argparse
import sys
import numpy as np  # 新增依赖
import torch       # 新增依赖
from rouge import Rouge
from bert_score import score as bert_score_func # 新增依赖

def load_outputs(file_path):
    """
    从 jsonl 文件加载输出。
    返回一个字典: key="client_id-round_idx", value=output_text
    """
    outputs = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                data = json.loads(line)
                
                # 构造唯一键，确保对比的是同一个请求的回复
                key = f"{data['client_id']}-{data.get('round_idx', 0)}"
                outputs[key] = data['output_text']
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {file_path}")
        sys.exit(1)
    return outputs

def main():
    parser = argparse.ArgumentParser(description="对比两次 Benchmark 的输出质量 (ROUGE-L + BERTScore)")
    parser.add_argument("--baseline", type=str, required=True, help="Baseline (关闭 Delta Cache) 的结果文件 (.jsonl)")
    parser.add_argument("--delta", type=str, required=True, help="Delta Cache (开启 Delta Cache) 的结果文件 (.jsonl)")
    args = parser.parse_args()

    print(f"正在加载 Baseline: {args.baseline} ...")
    base_map = load_outputs(args.baseline)
    
    print(f"正在加载 Delta:    {args.delta} ...")
    delta_map = load_outputs(args.delta)

    # =========================================================================
    # [新增逻辑] 批量计算 BERTScore (为了保证速度，在循环打印前统一计算)
    # =========================================================================
    print("\n⏳ 正在准备 BERTScore 计算 (这可能需要一点时间加载模型)...")
    
    # 预先对齐数据
    sorted_keys = sorted(base_map.keys(), key=lambda k: [int(x) for x in k.split('-')])
    bert_refs = []
    bert_cands = []
    valid_bert_keys = []

    for key in sorted_keys:
        if key in delta_map:
            ref = base_map[key]
            hyp = delta_map[key]
            # 过滤掉双空的情况 (双空稍后直接视为 1.0)
            if ref.strip() or hyp.strip():
                # BERTScore 不接受空字符串输入，如果一方为空，补一个空格
                bert_refs.append(ref if ref.strip() else " ")
                bert_cands.append(hyp if hyp.strip() else " ")
                valid_bert_keys.append(key)

    bert_scores_map = {}
    if bert_refs:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   需计算样本数: {len(bert_refs)} (Device: {device})")
        try:
            # 计算 F1 Score
            P, R, F1 = bert_score_func(
                bert_cands, bert_refs, 
                lang="en", 
                verbose=True, 
                device=device,
                batch_size=32
            )
            f1_list = F1.tolist()
            # 映射回 Key
            for i, key in enumerate(valid_bert_keys):
                bert_scores_map[key] = f1_list[i]
        except Exception as e:
            print(f"❌ BERTScore 计算失败: {e}")
    # =========================================================================

    rouge = Rouge()
    
    scores = []
    bert_scores_list = [] # 新增用于统计
    perfect_matches = 0
    mismatches = 0
    missing_keys = 0

    print("\n" + "="*100)
    # 调整表头宽度以容纳新指标
    print(f"{'ID-Round':<12} | {'Match?':<8} | {'ROUGE-L':<8} | {'BERT-F1':<8} | {'Quality Check'}")
    print("-" * 100)

    # 开始原有的逐行打印逻辑
    for key in sorted_keys:
        if key not in delta_map:
            missing_keys += 1
            continue
            
        ref_text = base_map[key]  # Baseline 输出 (参考答案)
        hyp_text = delta_map[key] # Delta 输出 (预测答案)
        
        # 1. 计算 ROUGE-L
        if not ref_text.strip() or not hyp_text.strip():
            score = 1.0 if ref_text == hyp_text else 0.0
        else:
            try:
                score_res = rouge.get_scores(hyp_text, ref_text)
                score = score_res[0]['rouge-l']['f']
            except Exception as e:
                print(f"⚠️ 计算出错: {e}")
                score = 0.0
        scores.append(score)

        # [新增] 获取 BERTScore
        if not ref_text.strip() and not hyp_text.strip():
            b_score = 1.0 # 双空视为完美
        else:
            b_score = bert_scores_map.get(key, 0.0)
        bert_scores_list.append(b_score)

        # 2. 状态判定
        is_perfect = (ref_text == hyp_text)
        if is_perfect:
            perfect_match_icon = "✅ SAME"
            perfect_matches += 1
        else:
            perfect_match_icon = "⚠️ DIFF"
            mismatches += 1
        
        # 3. 质量评级 (加入语义判定)
        if score > 0.99: quality = "Excellent"
        elif b_score > 0.98: quality = "Sem.Same" # 语义几乎一样
        elif score > 0.90: quality = "Good"
        elif b_score > 0.90: quality = "Sem.Good" # 语义不错
        elif score > 0.75: quality = "Acceptable"
        else: quality = "❌ POOR"

        # 打印行 (加入 b_score)
        print(f"{key:<12} | {perfect_match_icon:<8} | {score:.4f}   | {b_score:.4f}   | {quality}")

        # 4. 如果分数太低，打印对比以便调试
        if score < 0.95 and not is_perfect:
            print(f"   [Base]: {ref_text[:60].replace(chr(10), ' ')}...")
            print(f"   [Diff]: {hyp_text[:60].replace(chr(10), ' ')}...")

    # --- 最终统计 ---
    avg_rouge = sum(scores) / len(scores) if scores else 0
    avg_bert = np.mean(bert_scores_list) if bert_scores_list else 0.0 # 计算平均 BERTScore

    print("="*100)
    print("📊 最终测试报告")
    print(f"对比样本数: {len(scores)}")
    print(f"缺失样本数: {missing_keys}")
    print(f"完全一致率: {perfect_matches} / {len(scores)} ({perfect_matches/len(scores):.2%})")
    print(f"平均 ROUGE-L:   {avg_rouge:.5f}")
    print(f"平均 BERTScore: {avg_bert:.5f}  (语义相似度 - 核心指标)")
    print("-" * 100)
    
    # 结论判定优先参考 BERTScore
    if avg_bert > 0.99:
        print("结论: 🟢 Delta Cache 精度极高，语义近乎无损。")
    elif avg_bert > 0.95:
        print("结论: 🟡 精度有轻微损失 (措辞微调)，但在可接受范围内。")
    else:
        print("结论: 🔴 精度损失严重，建议检查压缩参数 (eb_abs) 或重建逻辑。")

if __name__ == "__main__":
    main()