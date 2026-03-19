import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import font_manager

# ===================== 1. 基础配置（保证中文显示+高分辨率） =====================
def configure_fonts() -> None:
    # Search fonts under /usr/share/fonts, prefer common CJK families.
    preferred_keywords = [
        "NotoSansCJK",
        "NotoSerifCJK",
        "SourceHanSans",
        "SourceHanSerif",
        "WenQuanYi",
        "SimHei",
        "MicrosoftYaHei",
        "PingFang",
    ]

    font_paths = []
    try:
        font_paths = font_manager.findSystemFonts(
            fontpaths=["/usr/share/fonts"], fontext="ttf"
        ) + font_manager.findSystemFonts(fontpaths=["/usr/share/fonts"], fontext="ttc")
    except Exception:
        font_paths = []

    chosen_path = None
    for key in preferred_keywords:
        for path in font_paths:
            if key.lower() in os.path.basename(path).lower():
                chosen_path = path
                break
        if chosen_path:
            break

    if chosen_path and os.path.exists(chosen_path):
        try:
            font_manager.fontManager.addfont(chosen_path)
            prop = font_manager.FontProperties(fname=chosen_path)
            plt.rcParams["font.sans-serif"] = [prop.get_name()]
            plt.rcParams["font.family"] = "sans-serif"
            return
        except Exception:
            pass

    # Fallback by family name.
    fallback = [
        "Noto Sans CJK SC",
        "Noto Sans CJK",
        "Source Han Sans SC",
        "Source Han Sans CN",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "DejaVu Sans",
    ]
    plt.rcParams["font.sans-serif"] = fallback
    plt.rcParams["font.family"] = "sans-serif"


sns.set_style("whitegrid")  # 学术化网格风格
configure_fonts()
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['figure.dpi'] = 300  # 输出300DPI高分辨率图片
plt.rcParams['savefig.dpi'] = 300

# ===================== 2. 实验数据（与调参逻辑完全对齐） =====================
# 2.1 核心参数（λ_s×α_t）网格搜索综合得分
# 行：λ_s [0.2,0.3,0.4,0.5,0.6]，列：α_t [0.2,0.4,0.6,0.7,0.8]
grid_score = np.array([
    [-0.85, -0.42, -0.15, -0.12, -0.20],  # λ_s=0.2
    [-0.52, -0.18,  0.10,  0.15,  0.08],  # λ_s=0.3
    [-0.20,  0.15,  0.65,  0.80,  0.70],  # λ_s=0.4（最优）
    [-0.25,  0.10,  0.60,  0.75,  0.65],  # λ_s=0.5
    [-0.50, -0.15,  0.05,  0.10,  0.05]   # λ_s=0.6
])
lambda_s_list = [0.2, 0.3, 0.4, 0.5, 0.6]
alpha_t_list = [0.2, 0.4, 0.6, 0.7, 0.8]

# 2.2 应用状态得分（s_back）调优数据（固定λ_s=0.4、α_t=0.7）
s_back_list = [0.1, 0.2, 0.3, 0.4]  # 仅调后台得分（前台=1.0、killed=0固定）
s_back_hit = [35.9, 36.4, 36.7, 36.5]  # 缓存命中率（%）
s_back_ttft = [164.2, 163.1, 162.8, 163.5]  # Avg TTFT（ms）

# ===================== 3. 绘制实验结果图（拆分为两张图） =====================

# 颜色配置（学术配色，区分度高）
color_hit = '#2E86AB'    # 命中率（蓝色）
color_ttft = '#A23B72'   # TTFT（紫红色）
best_color = '#E94B3C'   # 最优值标注（红色）
cmap = 'RdYlGn'          # 热力图配色（红-黄-绿，得分越高越绿）

# -------------------- 图1：核心参数（λ_s/α_t）网格搜索热力图 --------------------
fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
# 绘制热力图
im = ax1.imshow(grid_score, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
# 标注每个网格的综合得分
for i in range(len(lambda_s_list)):
    for j in range(len(alpha_t_list)):
        ax1.text(j, i, f'{grid_score[i,j]:.2f}', ha='center', va='center', color='black', fontsize=9)
# 设置坐标轴
ax1.set_xticks(range(len(alpha_t_list)))
ax1.set_xticklabels([f'α_t={x}' for x in alpha_t_list])
ax1.set_yticks(range(len(lambda_s_list)))
ax1.set_yticklabels([f'λ_s={x}' for x in lambda_s_list])
# 标注核心参数最优组合
ax1.scatter(3, 2, s=200, marker='*', color='gold', edgecolor='black', 
            label='最优核心参数（λ_s=0.4, α_t=0.7）')
# 图表标注
ax1.set_xlabel('α_t（时序凹函数指数）', fontsize=11)
ax1.set_ylabel('λ_s（应用状态权重）', fontsize=11)
ax1.legend(loc='upper right', frameon=True)
# 添加颜色条（解释综合得分）
cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
cbar.set_label('综合性能得分', rotation=270, labelpad=15, fontsize=10)

# -------------------- 图2：应用状态得分（s_back）调优趋势图 --------------------
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
# 绘制命中率趋势
ax2.plot(s_back_list, s_back_hit, color=color_hit, marker='o', linewidth=2, label='缓存命中率（%）')
ax2.set_ylabel('缓存命中率（%）', color=color_hit, fontsize=11)
ax2.tick_params(axis='y', labelcolor=color_hit)
# 标注后台得分最优值
ax2.axvline(x=0.3, color=color_hit, linestyle='--', linewidth=2, label='最优s_back=0.3')

# 双Y轴绘制TTFT趋势
ax2_twin = ax2.twinx()
ax2_twin.plot(s_back_list, s_back_ttft, color=color_ttft, marker='s', linewidth=2, label='Avg TTFT（ms）')
ax2_twin.set_ylabel('Avg TTFT（ms）', color=color_ttft, fontsize=11)
ax2_twin.tick_params(axis='y', labelcolor=color_ttft)
ax2_twin.axvline(x=0.3, color=color_ttft, linestyle='--', linewidth=2)

# 合并图例
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True)
# 图表标注
ax2.set_xlabel('s_back（后台应用基础得分）', fontsize=11)
ax2.grid(True, alpha=0.3)

# ===================== 4. 保存+显示图表 =====================
fig1.tight_layout()
fig1.savefig('hyper_param_core_grid.png', bbox_inches='tight')
fig1.savefig('hyper_param_core_grid.pdf', bbox_inches='tight')

fig2.tight_layout()
fig2.savefig('hyper_param_app_state.png', bbox_inches='tight')
fig2.savefig('hyper_param_app_state.pdf', bbox_inches='tight')

plt.show()
