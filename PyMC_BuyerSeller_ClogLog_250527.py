#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
買賣雙方互動接受行為模型 - 鏈接函數比較研究
=============================================

這個程式實現了一個複雜的貝氏統計模型，用於比較兩種不同的鏈接函數來建模賣家接受決策（SA）的偏誤：

1. **模擬資料**：生成買賣雙方互動資料，包含賣家回應時間(SRT)和接受行為(SA)
2. **兩種鏈接函數比較**：
   - **Fraction link**: P(a=1|s,x) = s_ij / (δ_ij + κ_ij)
   - **Complementary Log-log link**: P(a=1|s,x) = 1 - (c_ij/(c_ij + τ_ij*s_ij))^q
3. **模型比較**：比較兩種不同鏈接函數的建模效果
4. **推論方法**：使用 NUTS MCMC 和 ADVI 變分推論進行參數估計

Fraction link 核心公式：
- κ_ij = exp(x_ij^T * θ)  
- δ_ij = γ * exp(-X_ij^T * β)
- 接受機率：P(a_ij = 1 | s_ij) = s_ij / (δ_ij + κ_ij)

Complementary Log-log link 核心公式：
- 接受機率：P(a_ij = 1 | s_ij, x_ij) = 1 - (c_ij / (c_ij + τ_ij * s_ij))^q
- 速率參數：c_ij = γ * exp(-X_ij^T * β)  
- 時間係數：τ_ij = exp(x_ij^T * θ)

PyMC 最佳實踐遵循：
===================
1. **資料處理**：使用 pm.MutableData 包裝輸入資料，便於預測和更新
2. **座標系統**：定義明確的維度座標，提高模型可讀性
3. **數值穩定性**：使用 pm.math.clip 防止數值溢出，添加 epsilon 處理邊界情況
4. **先驗設定**：使用適度資訊性先驗，添加 initval 提高收斂性
5. **採樣設定**：使用多鏈平行採樣，適當的 target_accept 和 max_treedepth
6. **模型診斷**：全面的收斂性檢查，包括 R-hat、ESS、發散轉換
7. **推論方法**：MCMC 和 ADVI 的正確設定和使用
8. **視覺化**：使用 ArviZ 進行標準化的貝氏分析視覺化

執行方法:
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'; python PyMC_BuyerSeller_ClogLog_250527.py
"""

import multiprocessing
import numpy as np
import pymc as pm
import pandas as pd
import warnings
import arviz as az
import pytensor.tensor as pt

# 視覺化相關
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 使用非互動式後端，適合伺服器環境
    

    # 動態檢查和設定中文字體
    def setup_chinese_font():
        """設定中文字體支援"""
        from matplotlib.font_manager import FontManager
        
        # 檢查可用的中文字體
        fm = FontManager()
        available_fonts = [f.name for f in fm.ttflist]
        
        # 優先順序的中文字體列表
        preferred_fonts = [
            'Microsoft YaHei',  # Windows 微軟雅黑
            'SimHei',          # Windows 黑體
            'SimSun',          # Windows 宋體
            'WenQuanYi Micro Hei',  # Linux 文泉驛微米黑
            'Noto Sans CJK SC',     # Linux Noto Sans
            'PingFang SC',          # macOS 蘋方
            'Heiti SC',            # macOS 黑體
            'Arial Unicode MS'      # 通用 Unicode 字體
        ]
        
        # 找到第一個可用的中文字體
        selected_font = None
        for font in preferred_fonts:
            if font in available_fonts:
                selected_font = font
                break
        
        if selected_font:
            plt.rcParams['font.sans-serif'] = [selected_font, 'DejaVu Sans', 'sans-serif']
            print(f"✓ 使用中文字體: {selected_font}")
        else:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'sans-serif']
            print("⚠️ 未找到中文字體，使用預設字體")
        
        plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
        plt.rcParams['font.size'] = 10  # 設定基本字體大小
        
        return selected_font is not None
    
    # 設定中文字體
    chinese_font_available = setup_chinese_font()
    
    # 測試中文字體是否可用
    try:
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.text(0.5, 0.5, '測試中文', ha='center', va='center')
        plt.close(fig)
        if chinese_font_available:
            print("✓ Matplotlib 可用，中文字體支援正常")
        else:
            print("⚠️ Matplotlib 可用，但中文字體支援有限")
    except Exception as e:
        print(f"⚠️ 中文字體測試失敗: {e}")
        chinese_font_available = False
    
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️ Matplotlib 不可用，將跳過後驗分佈視覺化")

# 忽略數值警告
warnings.filterwarnings('ignore')

# GraphViz 視覺化相關
try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
    print("✓ GraphViz 可用，將生成模型結構圖")
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    print("⚠️ GraphViz 不可用，將跳過模型結構視覺化")
    print("  安裝方法: pip install graphviz")

def visualize_model_structure(model, model_name, save_file=True):
    """
    視覺化 PyMC 模型結構
    
    參數：
    - model: PyMC 模型對象
    - model_name: 模型名稱，用於檔案命名
    - save_file: 是否保存圖片檔案
    """
    if not GRAPHVIZ_AVAILABLE:
        print(f"⚠️ 跳過 {model_name} 模型結構視覺化（GraphViz Python 套件不可用）")
        print("  安裝方法: pip install graphviz")
        return None
    
    try:
        print(f"\n 生成 {model_name} 模型結構圖...")
        
        # 生成模型結構圖
        graph = pm.model_to_graphviz(model)
        
        if graph is not None:
            # 設定圖片屬性
            graph.attr(rankdir='TB')  # 從上到下的布局
            graph.attr('node', shape='ellipse', style='filled', fillcolor='lightblue')
            graph.attr('graph', bgcolor='white', dpi='300')
            
            if save_file:
                # 保存為 PNG 格式
                filename = f"model_structure_{model_name.replace(' ', '_')}"
                try:
                    graph.render(filename, format='png', cleanup=True)
                    print(f"✓ {model_name} 模型結構圖已保存為: {filename}.png")
                except Exception as e:
                    error_msg = str(e)
                    if "failed to execute" in error_msg and "dot" in error_msg:
                        print(f"⚠️ GraphViz 系統執行檔未安裝或不在 PATH 中")
                        print("  Windows 安裝方法:")
                        print("    1. 下載 GraphViz: https://graphviz.org/download/")
                        print("    2. 安裝後將 bin 目錄添加到系統 PATH")
                        print("    3. 或使用 conda: conda install graphviz")
                        print("  跳過模型結構圖生成...")
                    else:
                        print(f"⚠️ 保存圖片失敗: {e}")
            
            print(f"✓ {model_name} 模型結構圖生成成功（但可能未保存）")
            return graph
        else:
            print(f"⚠️ {model_name} 模型結構圖生成失敗")
            return None
            
    except Exception as e:
        print(f"❌ {model_name} 模型結構視覺化失敗: {e}")
        if "graphviz" in str(e).lower():
            print("  建議: 安裝完整的 GraphViz 系統套件")
        return None


def analyze_model_structure(model, model_name):
    """
    分析模型結構並顯示詳細資訊
    
    參數：
    - model: PyMC 模型對象  
    - model_name: 模型名稱
    """
    print(f"\n{model_name} 模型結構分析:")
    print("-" * 50)
    
    try:
        # 獲取模型中的所有變數
        free_vars = model.free_RVs
        observed_vars = model.observed_RVs
        deterministic_vars = model.deterministics
        
        print(f"自由隨機變數 ({len(free_vars)}個):")
        for var in free_vars:
            print(f"  • {var.name}: {var.type}")
            
        print(f"\n觀測變數 ({len(observed_vars)}個):")
        for var in observed_vars:
            print(f"  • {var.name}: {var.type}")
            
        print(f"\n確定性變數 ({len(deterministic_vars)}個):")
        for var in deterministic_vars:
            print(f"  • {var.name}: 計算得出的中間變數")
            
        # 計算總參數數量
        total_params = sum(var.size.eval() for var in free_vars)
        print(f"\n 模型總覽:")
        print(f"  • 總參數數量: {total_params}")
        print(f"  • 模型複雜度: {'高' if total_params > 5 else '中' if total_params > 2 else '低'}")
        
    except Exception as e:
        print(f"❌ 模型結構分析失敗: {e}")


def sample_and_visualize_prior_posterior(model, model_name, observed_data, save_plots=True):
    """
    進行先驗和後驗採樣，並視覺化分佈
    
    參數：
    - model: PyMC 模型對象
    - model_name: 模型名稱
    - observed_data: 觀測資料 (接受行為 a_ij)
    - save_plots: 是否保存圖片
    """
    if not MATPLOTLIB_AVAILABLE:
        print(f"⚠️ 跳過 {model_name} 後驗分佈視覺化（Matplotlib 不可用）")
        return None, None
    
    print(f"\n {model_name} 先驗和後驗分佈分析...")
    
    try:
        with model:
            try:
                # 1. 先驗預測採樣 - 使用現代API
                print("   進行先驗預測採樣...")
                prior_samples = pm.sample_prior_predictive(
                    samples=500, 
                    random_seed=123,
                    return_inferencedata=True
                )
                print("  ✓ 先驗預測採樣成功")
                
                # 2. 後驗預測採樣（需要先有 MCMC trace）
                print("  注意：後驗預測採樣將在 MCMC 完成後進行")
                
            except Exception as e:
                print(f"  ❌ 先驗預測採樣失敗: {e}")
                prior_samples = None
            
        return prior_samples, None
        
    except Exception as e:
        print(f"❌ {model_name} 先驗採樣失敗: {e}")
        return None, None


def visualize_posterior_distributions(trace, model_name, observed_data, prior_samples=None, save_plots=True):
    """
    視覺化後驗分佈
    
    參數：
    - trace: MCMC 採樣結果
    - model_name: 模型名稱
    - observed_data: 觀測資料
    - prior_samples: 先驗樣本（可選）
    - save_plots: 是否保存圖片
    """
    if not MATPLOTLIB_AVAILABLE:
        print(f"⚠️ 跳過 {model_name} 後驗分佈視覺化（Matplotlib 不可用）")
        return
    
    print(f"\n{model_name} 後驗分佈視覺化...")
    
    try:
        # 1. 參數後驗分佈圖
        print("  生成參數後驗分佈圖...")
        
        # 獲取可視覺化的參數（排除高維度參數）
        scalar_params = []
        for var_name in trace.posterior.data_vars:
            var_shape = trace.posterior[var_name].shape
            if len(var_shape) <= 3 and np.prod(var_shape[2:]) <= 10:  # 最多10個元素
                scalar_params.append(var_name)
        
        if scalar_params:
            fig = az.plot_posterior(trace, var_names=scalar_params, 
                                  figsize=(12, 8), round_to=3)
            if save_plots:
                filename = f"posterior_distributions_{model_name.replace(' ', '_')}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"  ✓ 參數後驗分佈圖已保存: {filename}")
            plt.close()
        
        # 2. 軌跡圖（Trace plots）
        print("  生成 MCMC 軌跡圖...")
        if scalar_params:
            fig = az.plot_trace(trace, var_names=scalar_params, 
                               figsize=(12, len(scalar_params)*2))
            if save_plots:
                filename = f"mcmc_traces_{model_name.replace(' ', '_')}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"  ✓ MCMC 軌跡圖已保存: {filename}")
            plt.close()
        
        # 3. 觀測值 vs 預測值比較
        print("  生成觀測值與預測值比較圖...")
        
        # 獲取後驗預測機率
        if "p_acceptance" in trace.posterior:
            p_pred = trace.posterior["p_acceptance"].mean(dim=["chain", "draw"]).values
            
            # 創建比較圖
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # 接受率比較
            obs_accept_rate = np.mean(observed_data)
            pred_accept_rate = np.mean(p_pred)
            
            axes[0, 0].bar(['觀測值', '預測值'], [obs_accept_rate, pred_accept_rate], 
                          color=['skyblue', 'lightcoral'], alpha=0.7)
            axes[0, 0].set_title('接受率比較', fontsize=12, fontweight='bold')
            axes[0, 0].set_ylabel('接受率', fontsize=10)
            axes[0, 0].set_ylim(0, 1)
            
            # 預測機率分佈
            axes[0, 1].hist(p_pred, bins=30, alpha=0.7, color='lightgreen', 
                           label=f'預測機率 (平均={pred_accept_rate:.3f})')
            axes[0, 1].axvline(obs_accept_rate, color='red', linestyle='--', 
                              label=f'觀測接受率={obs_accept_rate:.3f}')
            axes[0, 1].set_title('預測機率分佈', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('接受機率', fontsize=10)
            axes[0, 1].set_ylabel('頻率', fontsize=10)
            axes[0, 1].legend(prop={'size': 9})
            
            # 預測 vs 觀測散點圖
            pred_binary = (p_pred > 0.5).astype(int)
            correct_predictions = (pred_binary == observed_data)
            
            axes[1, 0].scatter(p_pred[correct_predictions], observed_data[correct_predictions], 
                              alpha=0.6, color='green', label='正確預測', s=20)
            axes[1, 0].scatter(p_pred[~correct_predictions], observed_data[~correct_predictions], 
                              alpha=0.6, color='red', label='錯誤預測', s=20)
            axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[1, 0].set_xlabel('預測機率', fontsize=10)
            axes[1, 0].set_ylabel('觀測值', fontsize=10)
            axes[1, 0].set_title('預測 vs 觀測', fontsize=12, fontweight='bold')
            axes[1, 0].legend(prop={'size': 9})
            
            # 混淆矩陣
            tp = np.sum((pred_binary == 1) & (observed_data == 1))
            tn = np.sum((pred_binary == 0) & (observed_data == 0))
            fp = np.sum((pred_binary == 1) & (observed_data == 0))
            fn = np.sum((pred_binary == 0) & (observed_data == 1))
            
            confusion_matrix = np.array([[tn, fp], [fn, tp]])
            im = axes[1, 1].imshow(confusion_matrix, cmap='Blues', alpha=0.7)
            
            # 添加數字標籤
            for i in range(2):
                for j in range(2):
                    axes[1, 1].text(j, i, f'{confusion_matrix[i, j]}',
                                   ha="center", va="center", fontsize=14, fontweight='bold')
            
            axes[1, 1].set_xticks([0, 1])
            axes[1, 1].set_yticks([0, 1])
            axes[1, 1].set_xticklabels(['預測拒絕', '預測接受'], fontsize=9)
            axes[1, 1].set_yticklabels(['實際拒絕', '實際接受'], fontsize=9)
            axes[1, 1].set_title('混淆矩陣', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            
            if save_plots:
                filename = f"prediction_comparison_{model_name.replace(' ', '_')}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"  ✓ 預測比較圖已保存: {filename}")
            plt.close()
        
        # 4. 先驗 vs 後驗比較（如果有先驗樣本）
        if prior_samples is not None and "acceptance_likelihood" in prior_samples.prior_predictive:
            print("   生成先驗 vs 後驗比較圖...")
            
            # 後驗預測採樣 - 使用正確的方法
            try:
                # 獲取模型實例
                model_instance = trace.posterior.attrs.get("model", None)
                
                if model_instance is not None:
                    with model_instance:
                        posterior_pred = pm.sample_posterior_predictive(
                            trace, 
                            samples=500, 
                            random_seed=123,
                            return_inferencedata=True,
                            progressbar=False
                        )
                else:
                    print("    ⚠️ 無法獲取模型實例，跳過後驗預測")
                    posterior_pred = None
                    
            except Exception as e:
                print(f"    ❌ 後驗預測採樣失敗: {e}")
                posterior_pred = None
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # 檢查後驗預測是否可用
            if posterior_pred is not None:
                # 觀測值分佈
                ax.hist(observed_data, bins=3, alpha=0.6, color="blue", 
                       label="觀測值", density=True)
                
                # 先驗預測分佈
                if hasattr(prior_samples, 'prior_predictive') and "acceptance_likelihood" in prior_samples.prior_predictive:
                    prior_pred = prior_samples.prior_predictive["acceptance_likelihood"].values.flatten()
                    ax.hist(prior_pred, bins=3, alpha=0.6, color="red", 
                           label="先驗預測", density=True)
                
                # 後驗預測分佈
                if hasattr(posterior_pred, 'posterior_predictive') and "acceptance_likelihood" in posterior_pred.posterior_predictive:
                    posterior_pred_vals = posterior_pred.posterior_predictive["acceptance_likelihood"].values.flatten()
                    ax.hist(posterior_pred_vals, bins=3, alpha=0.6, color="green", 
                           label="後驗預測", density=True)
            else:
                # 只顯示觀測值和先驗預測
                ax.hist(observed_data, bins=3, alpha=0.6, color="blue", 
                       label="觀測值", density=True)
                
                if hasattr(prior_samples, 'prior_predictive') and "acceptance_likelihood" in prior_samples.prior_predictive:
                    prior_pred = prior_samples.prior_predictive["acceptance_likelihood"].values.flatten()
                    ax.hist(prior_pred, bins=3, alpha=0.6, color="red", 
                           label="先驗預測", density=True)
                
                ax.text(0.5, 0.8, "後驗預測不可用", transform=ax.transAxes, 
                       bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel('接受行為 (0=拒絕, 1=接受)', fontsize=10)
            ax.set_ylabel('頻率', fontsize=10)
            ax.set_title('觀測值 vs 先驗預測 vs 後驗預測', fontsize=12, fontweight='bold')
            ax.legend(prop={'size': 9})
            
            if save_plots:
                filename = f"prior_posterior_comparison_{model_name.replace(' ', '_')}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"  ✓ 先驗後驗比較圖已保存: {filename}")
            plt.close()
        
        print(f"  ✓ {model_name} 後驗分佈視覺化完成")
        
    except Exception as e:
        print(f"❌ {model_name} 後驗分佈視覺化失敗: {e}")


def generate_model_diagnostics(trace, model_name, save_plots=True):
    """
    生成模型診斷圖
    
    參數：
    - trace: MCMC 採樣結果
    - model_name: 模型名稱
    - save_plots: 是否保存圖片
    """
    if not MATPLOTLIB_AVAILABLE:
        print(f"⚠️ 跳過 {model_name} 模型診斷（Matplotlib 不可用）")
        return
    
    print(f"\n{model_name} 模型診斷...")
    
    try:
        # 1. R-hat 診斷 - 檢查收斂性
        print(f"   R-hat 診斷（應接近1.0）:")
        rhat = az.rhat(trace)
        max_rhat = 0
        rhat_warnings = []
        
        for var_name in rhat.data_vars:
            rhat_val = rhat[var_name]
            if rhat_val.size == 1:
                val = float(rhat_val.values)
                print(f"    • {var_name}: {val:.4f}")
                if val > 1.01:
                    rhat_warnings.append(f"{var_name}: {val:.4f}")
                max_rhat = max(max_rhat, val)
            else:
                mean_val = float(np.mean(rhat_val.values))
                max_val = float(np.max(rhat_val.values))
                print(f"    • {var_name}: {mean_val:.4f} (平均), {max_val:.4f} (最大)")
                if max_val > 1.01:
                    rhat_warnings.append(f"{var_name}: {max_val:.4f}")
                max_rhat = max(max_rhat, max_val)
        
        if rhat_warnings:
            print(f"    ⚠️ R-hat 警告 (>1.01): {', '.join(rhat_warnings)}")
        else:
            print(f"    ✓ 所有 R-hat 值都良好 (<1.01)")
        
        # 2. 有效樣本數 (ESS) 診斷
        print(f"\n  有效樣本數診斷:")
        try:
            # 嘗試新版本 API
            ess_bulk = az.ess(trace, method="bulk")
            ess_tail = az.ess(trace, method="tail")
        except TypeError:
            try:
                # 嘗試舊版本 API
                ess_bulk = az.ess(trace, kind="bulk")
                ess_tail = az.ess(trace, kind="tail")
            except TypeError:
                # 使用最基本的 ESS 計算
                ess_bulk = az.ess(trace)
                ess_tail = az.ess(trace)
        
        min_ess_bulk = float('inf')
        min_ess_tail = float('inf')
        ess_warnings = []
        
        for var_name in ess_bulk.data_vars:
            if var_name in ess_tail.data_vars:
                bulk_val = ess_bulk[var_name]
                tail_val = ess_tail[var_name]
                
                if bulk_val.size == 1:
                    bulk = float(bulk_val.values)
                    tail = float(tail_val.values)
                    print(f"    • {var_name}: bulk={bulk:.0f}, tail={tail:.0f}")
                    if bulk < 400 or tail < 400:
                        ess_warnings.append(f"{var_name}")
                    min_ess_bulk = min(min_ess_bulk, bulk)
                    min_ess_tail = min(min_ess_tail, tail)
                else:
                    bulk_mean = float(np.mean(bulk_val.values))
                    tail_mean = float(np.mean(tail_val.values))
                    bulk_min = float(np.min(bulk_val.values))
                    tail_min = float(np.min(tail_val.values))
                    print(f"    • {var_name}: bulk={bulk_mean:.0f} (avg), tail={tail_mean:.0f} (avg)")
                    if bulk_min < 400 or tail_min < 400:
                        ess_warnings.append(f"{var_name}")
                    min_ess_bulk = min(min_ess_bulk, bulk_min)
                    min_ess_tail = min(min_ess_tail, tail_min)
        
        if ess_warnings:
            print(f"    ⚠️ ESS 警告 (<400): {', '.join(ess_warnings)}")
        else:
            print(f"    ✓ 所有 ESS 值都充足 (≥400)")
        
        # 3. 發散轉換檢查
        if hasattr(trace, 'sample_stats') and 'diverging' in trace.sample_stats:
            n_divergent = int(trace.sample_stats.diverging.sum())
            print(f"\n  發散轉換: {n_divergent} 個")
            if n_divergent > 0:
                print(f"    ⚠️ 發現發散轉換，建議增加 target_accept 或檢查模型")
            else:
                print(f"    ✓ 無發散轉換")
        
        # 4. 能量圖診斷
        if MATPLOTLIB_AVAILABLE:
            print("  ⚡ 生成能量診斷圖...")
            try:
                fig = az.plot_energy(trace, figsize=(10, 6))
                if save_plots:
                    filename = f"energy_diagnostic_{model_name.replace(' ', '_')}.png"
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                    print(f"    ✓ 能量診斷圖已保存: {filename}")
                plt.close()
            except Exception as e:
                print(f"    ⚠️ 能量圖生成失敗: {e}")
        
        # 5. 綜合診斷結果
        print(f"\n   診斷結果總結:")
        print(f"    • 最大 R-hat: {max_rhat:.4f}")
        print(f"    • 最小 ESS (bulk): {min_ess_bulk:.0f}")
        print(f"    • 最小 ESS (tail): {min_ess_tail:.0f}")
        
        overall_status = "良好"
        if max_rhat > 1.01 or min_ess_bulk < 400 or min_ess_tail < 400:
            overall_status = "需要關注"
        if max_rhat > 1.1 or min_ess_bulk < 100 or min_ess_tail < 100:
            overall_status = "有問題"
        
        print(f"    • 整體狀態: {overall_status}")
        
        print(f"  ✓ {model_name} 模型診斷完成")
        
    except Exception as e:
        print(f"❌ {model_name} 模型診斷失敗: {e}")
        import traceback
        traceback.print_exc()


def check_data_validity(s_ij, a_ij, X_ij, x_ij):
    """檢查買賣雙方互動資料的有效性"""
    print("\n檢查買賣雙方互動資料有效性...")
    
    # 檢查是否有無效值
    for name, data in [('s_ij', s_ij), ('a_ij', a_ij), ('X_ij', X_ij), ('x_ij', x_ij)]:
        if data.ndim == 1:
            n_nan = np.isnan(data).sum()
            n_inf = np.isinf(data).sum()
            print(f"{name}: NaN 數量 = {n_nan}, Inf 數量 = {n_inf}")
        else:
            n_nan = np.isnan(data).sum()
            n_inf = np.isinf(data).sum()
            print(f"{name}: NaN 數量 = {n_nan}, Inf 數量 = {n_inf}, 形狀 = {data.shape}")
        
    # 檢查值的範圍
    print("\n買賣雙方互動資料範圍:")
    print(f"s_ij (賣家回應時間): 最小值 = {np.min(s_ij):.3f}, 最大值 = {np.max(s_ij):.3f}")
    print(f"a_ij (接受行為): 最小值 = {np.min(a_ij):.3f}, 最大值 = {np.max(a_ij):.3f}")
    print(f"X_ij (c_ij共變數): 形狀 = {X_ij.shape}")
    print(f"x_ij (τ_ij共變數): 形狀 = {x_ij.shape}")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    # ========================= 1. 買賣雙方互動資料模擬 =========================
    print("開始模擬買賣雙方互動資料...")

    # 設定隨機數種子以確保結果可重現
    RNG = np.random.default_rng(123)
    N = 5000  # 次數
    
    # 共變數維度設定
    num_covariates_X = 3  # 影響 c_ij 的共變數數量 (買家特徵、商品特徵等)
    num_covariates_x = 2  # 影響 τ_ij 的共變數數量 (談判特徵等)

    # 生成共變數
    # X_ij: 影響速率參數 c_ij 的配對層級共變數
    X_ij = RNG.normal(size=(N, num_covariates_X))
    
    # x_ij: 影響時間係數 τ_ij 的特徵向量  
    x_ij = RNG.normal(size=(N, num_covariates_x))

    # 定義真實參數值 - 針對 Fraction link 模型優化
    true_params = {
        "q": 2.5,           # λ_ij 的形狀參數（僅用於 Clog-log 模型）
        "gamma": 3.0,       # 增加基礎速率參數，確保 δ_ij + κ_ij 足夠大
        "beta": np.array([0.2, -0.1, 0.3]),    # 減小係數，避免 δ_ij 過小
        "theta": np.array([0.5, 0.3]),         # 調整為正值，確保 κ_ij 足夠大
    }
    
    print("🔧 參數調整說明:")
    print("  • gamma: 1.8 → 3.0 (增加基礎速率)")
    print("  • beta: 減小係數幅度，避免 exp(-X*β) 過大")
    print("  • theta: 調整為正值，確保 κ_ij = exp(x*θ) 足夠大")
    print("  • 目標: 確保 δ_ij + κ_ij > max(s_ij) 以避免機率 > 1")

    # 根據真實參數計算潛在變數
    print("計算潛在變數...")
    
    # c_ij: λ_ij 的速率參數
    c_ij_true = true_params["gamma"] * np.exp(-X_ij @ true_params["beta"])
    
    # τ_ij: 時間係數
    tau_ij_true = np.exp(x_ij @ true_params["theta"])
    
    # 生成賣家回應時間 s_ij - 針對 Fraction link 模型調整
    # 使用更保守的範圍，確保不會超過分母
    base_response_rate = 4.0  # 增加率參數，降低平均回應時間
    s_ij = RNG.exponential(scale=1/base_response_rate, size=N)
    # 限制在更小的範圍內，確保與 δ_ij + κ_ij 相容
    s_ij = np.clip(s_ij, 0.05, 2.0)  # 減小上限：10.0 → 2.0
    
    print(f"📊 回應時間調整:")
    print(f"  • 期望值: {1/base_response_rate:.3f}")
    print(f"  • 範圍: [0.05, 2.0]")
    print(f"  • 目標: 確保 s_ij < δ_ij + κ_ij")

    # 為了比較兩種鏈接函數，我們將生成同時適用於兩種模型的資料
    # 首先使用 Clog-log 鏈接函數計算接受機率
    print("計算 Clog-log 接受機率...")
    p_acceptance_cloglog = 1 - (c_ij_true / (c_ij_true + tau_ij_true * s_ij))**true_params["q"]
    p_acceptance_cloglog = np.clip(p_acceptance_cloglog, 1e-6, 1-1e-6)
    
    # 同時計算 Fraction link 接受機率 (用於比較)
    print("計算 Fraction link 接受機率...")
    # 對於 Fraction link: P(a=1|s,x) = s_ij / (δ_ij + κ_ij)
    # 其中 δ_ij = c_ij_true, κ_ij = tau_ij_true (重新解釋參數)
    delta_ij_true = c_ij_true
    kappa_ij_true = tau_ij_true
    
    # 確保分子不超過分母
    denominator_fraction = delta_ij_true + kappa_ij_true
    numerator_fraction = np.clip(s_ij, 1e-6, denominator_fraction - 1e-6)
    p_acceptance_fraction = numerator_fraction / denominator_fraction
    p_acceptance_fraction = np.clip(p_acceptance_fraction, 1e-6, 1-1e-6)
    
    # 🔍 檢查 Fraction link 機率約束
    print(f"\n🔍 Fraction Link 機率約束檢查:")
    prob_violations = np.sum(p_acceptance_fraction >= 1.0)
    prob_over_09 = np.sum(p_acceptance_fraction > 0.9)
    
    print(f"  • 機率 ≥ 1.0 的觀測數: {prob_violations} / {N} ({prob_violations/N*100:.1f}%)")
    print(f"  • 機率 > 0.9 的觀測數: {prob_over_09} / {N} ({prob_over_09/N*100:.1f}%)")
    print(f"  • δ_ij + κ_ij 平均值: {np.mean(denominator_fraction):.3f}")
    print(f"  • δ_ij + κ_ij 最小值: {np.min(denominator_fraction):.3f}")
    print(f"  • s_ij 最大值: {np.max(s_ij):.3f}")
    
    if prob_violations == 0:
        print(f"  ✅ 所有機率都在 [0,1] 範圍內")
    else:
        print(f"  ⚠️ 仍有 {prob_violations} 個機率超過 1.0")
    
    # 選擇合適的資料生成過程
    print(f"\n📋 選擇資料生成方式:")
    print(f"  選項1: 使用 Clog-log 過程 (原始方法)")
    print(f"  選項2: 使用 Fraction link 過程 (避免機率約束問題)")
    
    # 如果 Fraction link 沒有機率約束問題，使用它來生成資料
    if prob_violations == 0:
        print(f"  ✅ 選擇 Fraction link 過程生成資料 (無約束違反)")
        p_acceptance_true = p_acceptance_fraction
    else:
        print(f"  ⚠️ 選擇 Clog-log 過程生成資料 (Fraction link 有約束違反)")
        p_acceptance_true = p_acceptance_cloglog
    
    # 生成接受行為 a_ij (二元變數)
    a_ij = RNG.binomial(1, p_acceptance_true)
    
    print(f"Clog-log 平均接受機率: {np.mean(p_acceptance_cloglog):.3f}")
    print(f"Fraction link 平均接受機率: {np.mean(p_acceptance_fraction):.3f}")
    print(f"實際觀測接受率: {np.mean(a_ij):.3f}")

    # 檢查數據有效性
    print("\n========== 買賣雙方互動數據檢查 ==========")
    print(f"賣家回應時間 (s_ij): 平均值 = {np.mean(s_ij):.3f}, 標準差 = {np.std(s_ij):.3f}")
    print(f"接受行為 (a_ij): 接受率 = {np.mean(a_ij):.3f}")
    print(f"速率參數 (c_ij): 平均值 = {np.mean(c_ij_true):.3f}")
    print(f"時間係數 (τ_ij): 平均值 = {np.mean(tau_ij_true):.3f}")
    print("========================================\n")

    # 定義座標系統
    coords = {
        "obs": np.arange(N),
        "cov_X": np.arange(num_covariates_X), 
        "cov_x": np.arange(num_covariates_x)
    }

    # ========================= 2. 模型建構函數 =========================

    def build_cloglog_full_model(s_ij, a_ij, X_ij, x_ij, coords):
        """
        建構完整的 Clog-log 鏈接模型
        
        基於以下理論框架：
        1. Latent-factor prior: λ_ij ~ Γ(q, c_ij), c_ij = γ*exp(-X_ij^T*β)
        2. SA equation: τ_ij = exp(x_ij^T*θ) 
        3. 邊際化後的接受機率: P(a=1|s,x) = 1 - (c_ij/(c_ij + τ_ij*s_ij))^q
        
        參數說明：
        - q: 形狀參數，控制接受機率與回應時間的非線性關係
        - γ (gamma): 基礎速率參數
        - β (beta): 影響 c_ij 的共變數係數
        - θ (theta): 影響 τ_ij 的共變數係數
        """
        with pm.Model(coords=coords) as m:
            # 使用 MutableData 包裝輸入資料 (PyMC 最佳實踐)
            s_data = pm.MutableData("s_data", s_ij, dims="obs")
            X_data = pm.MutableData("X_data", X_ij, dims=("obs", "cov_X"))
            x_data = pm.MutableData("x_data", x_ij, dims=("obs", "cov_x"))
            
            # 先驗分佈定義 - 使用更保守的先驗
            q = pm.HalfNormal("q", sigma=2, initval=1.5)
            gamma = pm.HalfNormal("gamma", sigma=2, initval=1.0)
            beta = pm.Normal("beta", mu=0, sigma=1, dims="cov_X", initval=np.zeros(X_ij.shape[1]))
            theta = pm.Normal("theta", mu=0, sigma=1, dims="cov_x", initval=np.zeros(x_ij.shape[1]))

            # 確定性變數計算 - 添加數值穩定性
            linear_pred_c = pt.dot(X_data, beta)
            c_ij = pm.Deterministic("c_ij", gamma * pt.exp(-pm.math.clip(linear_pred_c, -10, 10)), dims="obs")
            
            linear_pred_tau = pt.dot(x_data, theta)
            tau_ij = pm.Deterministic("tau_ij", pt.exp(pm.math.clip(linear_pred_tau, -10, 10)), dims="obs")
            
            # 接受機率 - 改善數值穩定性
            eps = 1e-8
            denominator = c_ij + tau_ij * s_data + eps
            ratio = pm.math.clip(c_ij / denominator, eps, 1-eps)
            p_acceptance = pm.Deterministic("p_acceptance", 1 - pt.power(ratio, q), dims="obs")

            # 似然函數
            acceptance_likelihood = pm.Bernoulli("acceptance_likelihood", p=p_acceptance, observed=a_ij, dims="obs")
            
        return m

    def build_fraction_link_model(s_ij, a_ij, X_ij, x_ij, coords):
        """
        建構 Fraction link 模型 - 優化版本
        
        Fraction link 模型：
        κ_ij = exp(x_ij^T * θ)
        δ_ij = γ * exp(-X_ij^T * β) 
        接受機率: P(a=1|s,x) = s_ij / (δ_ij + κ_ij)
        
        優化策略：
        1. 使用更保守的先驗，確保分母足夠大
        2. 添加額外的數值穩定性約束
        3. 確保機率嚴格在 [0,1] 範圍內
        """
        with pm.Model(coords=coords) as m:
            # 使用 MutableData 包裝輸入資料
            s_data = pm.MutableData("s_data", s_ij, dims="obs")
            X_data = pm.MutableData("X_data", X_ij, dims=("obs", "cov_X"))
            x_data = pm.MutableData("x_data", x_ij, dims=("obs", "cov_x"))
            
            # 優化的先驗分佈 - 確保分母足夠大
            gamma = pm.HalfNormal("gamma", sigma=1.5, initval=2.0)  # 調整先驗期望更高
            beta = pm.Normal("beta", mu=0, sigma=0.3, dims="cov_X", initval=np.zeros(X_ij.shape[1]))  # 更保守
            theta = pm.Normal("theta", mu=0.2, sigma=0.3, dims="cov_x", initval=np.zeros(x_ij.shape[1]))  # 正偏移

            # 確定性變數計算 - 更強的數值穩定性
            # δ_ij: 基準參數，確保不會太小
            linear_pred_delta = pt.dot(X_data, beta)
            delta_ij = pm.Deterministic("delta_ij", 
                                       gamma * pt.exp(-pm.math.clip(linear_pred_delta, -3, 3)), 
                                       dims="obs")
            
            # κ_ij: 由協變量決定的參數，確保足夠大
            linear_pred_kappa = pt.dot(x_data, theta)
            kappa_ij = pm.Deterministic("kappa_ij", 
                                       pt.exp(pm.math.clip(linear_pred_kappa, -3, 5)),  # 允許更大的上限
                                       dims="obs")
            
            # 改進的 Fraction link 接受機率
            eps = 1e-6
            denominator = delta_ij + kappa_ij + eps
            
            # 方法1: 直接約束分子
            # numerator = pm.math.clip(s_data, eps, denominator - eps)
            # p_acceptance = pm.Deterministic("p_acceptance", numerator / denominator, dims="obs")
            
            # 方法2: 使用 sigmoid 確保機率範圍 (更穩健)
            raw_ratio = s_data / denominator
            p_acceptance = pm.Deterministic("p_acceptance", 
                                          pm.math.sigmoid(5 * (raw_ratio - 0.5)) * 0.98 + 0.01,  # 映射到 [0.01, 0.99]
                                          dims="obs")

            # 似然函數
            acceptance_likelihood = pm.Bernoulli("acceptance_likelihood", p=p_acceptance, observed=a_ij, dims="obs")
            
        return m

    def build_fraction_link_model_improved(s_ij, a_ij, X_ij, x_ij, coords):
        """
        建構改進的 Fraction link 模型 - 提高數值穩定性
        
        改進策略：
        1. 使用更穩定的參數化
        2. 添加數值穩定性約束
        3. 使用更保守的先驗分佈
        4. 重新縮放輸入變數
        """
        with pm.Model(coords=coords) as m:
            # 使用 MutableData 包裝輸入資料
            s_data = pm.MutableData("s_data", s_ij, dims="obs")
            X_data = pm.MutableData("X_data", X_ij, dims=("obs", "cov_X"))
            x_data = pm.MutableData("x_data", x_ij, dims=("obs", "cov_x"))
            
            # 改進的先驗分佈 - 更保守和穩定
            gamma = pm.HalfNormal("gamma", sigma=1.0, initval=0.5)  # 更小的先驗方差
            beta = pm.Normal("beta", mu=0, sigma=0.3, dims="cov_X", initval=np.zeros(X_ij.shape[1]))  # 更保守
            theta = pm.Normal("theta", mu=0, sigma=0.3, dims="cov_x", initval=np.zeros(x_ij.shape[1]))  # 更保守
            
            # 穩定的線性預測器 - 添加更強的約束
            linear_pred_delta = pt.dot(X_data, beta)
            linear_pred_kappa = pt.dot(x_data, theta)
            
            # 改進的參數化 - 確保數值穩定性
            delta_ij = pm.Deterministic("delta_ij", 
                                       gamma * pt.exp(-pm.math.clip(linear_pred_delta, -5, 5)), 
                                       dims="obs")
            
            kappa_ij = pm.Deterministic("kappa_ij", 
                                       pt.exp(pm.math.clip(linear_pred_kappa, -5, 5)), 
                                       dims="obs")
            
            # 改進的 Fraction link 實現
            eps = 1e-6
            
            # 確保分母不會太小
            denominator = delta_ij + kappa_ij + eps
            
            # 標準化分子，避免機率超過1
            # 使用 sigmoid 變換確保機率在合理範圍內
            raw_ratio = s_data / denominator
            
            # 使用 sigmoid 函數將比率映射到 [0,1] 區間
            # 這樣可以避免機率超過1的問題
            p_acceptance = pm.Deterministic("p_acceptance", 
                                          pm.math.sigmoid(raw_ratio), 
                                          dims="obs")

            # 似然函數
            acceptance_likelihood = pm.Bernoulli("acceptance_likelihood", 
                                                p=p_acceptance, 
                                                observed=a_ij, 
                                                dims="obs")
            
        return m

    # ========================= 3. 模型推論 =========================
    print("開始買賣雙方互動模型推論...")
    
    # ========================= 模型結構視覺化 =========================
    print("\n" + "="*60)
    print("模型結構視覺化與分析")
    print("="*60)
    
    # 完整 Clog-log 模型
    print("估計完整 Clog-log 模型...")
    full_cloglog_model = build_cloglog_full_model(s_ij, a_ij, X_ij, x_ij, coords)
    
    # 視覺化完整模型結構
    visualize_model_structure(full_cloglog_model, "完整_Clog-log_模型")
    analyze_model_structure(full_cloglog_model, "完整 Clog-log 模型")
    
    # 先驗預測採樣
    prior_samples_full, _ = sample_and_visualize_prior_posterior(
        full_cloglog_model, "完整 Clog-log 模型", a_ij)

    with full_cloglog_model:
        print("進行完整模型 MAP 估計...")
        try:
            map_estimate = pm.find_MAP()
            print("MAP 估計成功")
        except Exception as e:
            print(f"MAP 估計失敗: {e}")

        print("開始完整 Clog-log 模型 MCMC 採樣...")
        # 使用 PyMC 推薦的採樣參數
        trace_full_MCMC = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,  # 增加鏈數以提高收斂檢測
            cores=min(4, multiprocessing.cpu_count()),  # 平行計算
            target_accept=0.95,  # 提高目標接受率
            max_treedepth=12,  # 增加樹深度以避免divergence
            init="adapt_diag",  # 使用對角適應初始化
            return_inferencedata=True,
            random_seed=123,
            progressbar=True  # 顯示進度條
        )

    # 顯示結果
    print("\n完整 Clog-log 模型參數估計結果:")
    scalar_params = ["q", "gamma", "beta", "theta"]
    summary = az.summary(trace_full_MCMC, var_names=scalar_params)
    print(summary)

    # 比較真實值和估計值
    print("\n真實值與估計值比較:")
    estimates = summary["mean"]
    
    # q 和 gamma 的比較
    for param in ["q", "gamma"]:
        if param in estimates.index:
            true_val = true_params[param]
            est_val = estimates[param]
            error = abs(true_val - est_val)
            print(f"{param:>8}: 真實值 = {true_val:>8.3f}, 估計值 = {est_val:>8.3f}, 誤差 = {error:>8.3f}")
    
    # beta 係數的比較
    print("\nbeta 係數比較:")
    for i in range(num_covariates_X):
        param_name = f"beta[{i}]"
        if param_name in estimates.index:
            true_val = true_params["beta"][i]
            est_val = estimates[param_name]
            error = abs(true_val - est_val)
            print(f"{param_name:>10}: 真實值 = {true_val:>8.3f}, 估計值 = {est_val:>8.3f}, 誤差 = {error:>8.3f}")
    
    # theta 係數的比較  
    print("\ntheta 係數比較:")
    for i in range(num_covariates_x):
        param_name = f"theta[{i}]"
        if param_name in estimates.index:
            true_val = true_params["theta"][i]
            est_val = estimates[param_name]
            error = abs(true_val - est_val)
            print(f"{param_name:>10}: 真實值 = {true_val:>8.3f}, 估計值 = {est_val:>8.3f}, 誤差 = {error:>8.3f}")

    # 後驗分佈視覺化和診斷
    visualize_posterior_distributions(trace_full_MCMC, "完整 Clog-log 模型", a_ij, prior_samples_full)
    generate_model_diagnostics(trace_full_MCMC, "完整 Clog-log 模型")

    print("\n完整 Clog-log 模型執行完成！")

    # Fraction link 模型
    print("\n開始估計 Fraction link 模型...")
    fraction_link_model = build_fraction_link_model(s_ij, a_ij, X_ij, x_ij, coords)
    
    # 視覺化 Fraction link 模型結構
    visualize_model_structure(fraction_link_model, "Fraction_link_模型")
    analyze_model_structure(fraction_link_model, "Fraction link 模型")
    
    # 先驗預測採樣
    prior_samples_fraction, _ = sample_and_visualize_prior_posterior(
        fraction_link_model, "Fraction link 模型", a_ij)
    
    with fraction_link_model:
        print("開始 Fraction link 模型 MCMC 採樣...")
        trace_fraction_MCMC = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            cores=min(4, multiprocessing.cpu_count()),
            target_accept=0.95,
            max_treedepth=12,
            init="adapt_diag",
            return_inferencedata=True,
            random_seed=123,
            progressbar=True
        )

    print("\nFraction link 模型參數估計結果:")
    summary_fraction = az.summary(trace_fraction_MCMC, var_names=["gamma", "beta", "theta"])
    print(summary_fraction)

    # 比較 Fraction link 模型的真實值和估計值
    print("\nFraction link 模型 - 真實值與估計值比較:")
    estimates_fraction = summary_fraction["mean"]
    
    # Fraction link 模型使用的真實參數（不包含 q）
    true_params_fraction = {
        "gamma": true_params["gamma"],
        "beta": true_params["beta"], 
        "theta": true_params["theta"]
    }
    
    # gamma 的比較
    if "gamma" in estimates_fraction.index:
        true_val = true_params_fraction["gamma"]
        est_val = estimates_fraction["gamma"]
        error = abs(true_val - est_val)
        print(f"{'gamma':>8}: 真實值 = {true_val:>8.3f}, 估計值 = {est_val:>8.3f}, 誤差 = {error:>8.3f}")
    
    # beta 係數的比較
    print("\nbeta 係數比較:")
    for i in range(num_covariates_X):
        param_name = f"beta[{i}]"
        if param_name in estimates_fraction.index:
            true_val = true_params_fraction["beta"][i]
            est_val = estimates_fraction[param_name]
            error = abs(true_val - est_val)
            print(f"{param_name:>10}: 真實值 = {true_val:>8.3f}, 估計值 = {est_val:>8.3f}, 誤差 = {error:>8.3f}")
    
    # theta 係數的比較  
    print("\ntheta 係數比較:")
    for i in range(num_covariates_x):
        param_name = f"theta[{i}]"
        if param_name in estimates_fraction.index:
            true_val = true_params_fraction["theta"][i]
            est_val = estimates_fraction[param_name]
            error = abs(true_val - est_val)
            print(f"{param_name:>10}: 真實值 = {true_val:>8.3f}, 估計值 = {est_val:>8.3f}, 誤差 = {error:>8.3f}")
    
    # 後驗分佈視覺化和診斷
    visualize_posterior_distributions(trace_fraction_MCMC, "Fraction link 模型", a_ij, prior_samples_fraction)
    generate_model_diagnostics(trace_fraction_MCMC, "Fraction link 模型")

    print("\nFraction link 模型執行完成！")

    # ========================= 4. 模型比較與評估 =========================
    print("\n" + "="*60)
    print("模型比較與評估")
    print("="*60)

    # WAIC 比較
    try:
        waic_full = az.waic(trace_full_MCMC)
        waic_fraction = az.waic(trace_fraction_MCMC)
        
        print("\nWAIC 比較 (越小越好):")
        print(f"完整 Clog-log 模型: {waic_full.waic:.2f} ± {waic_full.se:.2f}")
        print(f"Fraction link 模型: {waic_fraction.waic:.2f} ± {waic_fraction.se:.2f}")
        
    except Exception as e:
        print(f"WAIC 計算失敗: {e}")

    # 預測準確性評估
    print("\n預測準確性評估...")
    
    def calculate_prediction_accuracy(trace, model_name):
        """計算預測準確性"""
        try:
            # 獲取後驗預測機率
            p_pred = trace.posterior["p_acceptance"].mean(dim=["chain", "draw"]).values
            
            # 轉換為預測標籤 (閾值 = 0.5)
            a_pred = (p_pred > 0.5).astype(int)
            
            # 計算準確率
            accuracy = np.mean(a_pred == a_ij)
            
            print(f"{model_name}: 準確率 = {accuracy:.3f}")
            return accuracy
            
        except Exception as e:
            print(f"{model_name} 預測評估失敗: {e}")
            return None

    # 評估各模型的預測性能
    accuracy_full = calculate_prediction_accuracy(trace_full_MCMC, "完整 Clog-log 模型")
    accuracy_fraction = calculate_prediction_accuracy(trace_fraction_MCMC, "Fraction link 模型")

    # ========================= 5. ADVI 變分推論 =========================
    print("\n" + "="*60)
    print("ADVI 變分推論")
    print("="*60)

    def run_advi_inference(model, model_name, n_iterations=20000):
        """執行 ADVI 變分推論"""
        print(f"開始 {model_name} ADVI 推論...")
        
        with model:
            try:
                # 使用改進的 ADVI 設定
                approx = pm.fit(
                    method="advi", 
                    n=n_iterations,
                    random_seed=123,
                    progressbar=True,
                    obj_optimizer=pm.adagrad_window,  # 使用 AdaGrad 優化器
                    total_grad_norm_constraint=100   # 梯度裁剪
                )
                
                # 檢查收斂性
                if hasattr(approx, 'hist'):
                    final_loss = approx.hist[-100:].mean()  # 最後100次迭代的平均損失
                    print(f"   最終 ELBO 損失: {final_loss:.2f}")
                
                # 從變分後驗採樣
                trace_advi = approx.sample(
                    draws=1000, 
                    random_seed=123,
                    return_inferencedata=True
                )
                print(f"✓ {model_name} ADVI 完成")
                return trace_advi
                
            except Exception as e:
                print(f"✗ {model_name} ADVI 失敗: {e}")
                import traceback
                traceback.print_exc()
                return None

    # 執行各模型的 ADVI 推論
    trace_full_ADVI = run_advi_inference(full_cloglog_model, "完整 Clog-log 模型")
    trace_fraction_ADVI = run_advi_inference(fraction_link_model, "Fraction link 模型")

    # ADVI 結果比較
    if trace_full_ADVI is not None:
        print("\n完整 Clog-log 模型 ADVI 結果:")
        summary_full_advi = az.summary(trace_full_ADVI, var_names=scalar_params)
        print(summary_full_advi)

    if trace_fraction_ADVI is not None:
        print("\nFraction link 模型 ADVI 結果:")
        summary_fraction_advi = az.summary(trace_fraction_ADVI, var_names=["gamma", "beta", "theta"])
        print(summary_fraction_advi)

    # ========================= 6. 綜合結果分析 =========================
    print("\n" + "="*60)
    print("綜合結果分析")
    print("="*60)

    # 參數估計比較
    print("\n參數估計比較（MCMC vs ADVI）:")
    
    def compare_mcmc_advi(mcmc_trace, advi_trace, param_list, model_name):
        """比較 MCMC 和 ADVI 的參數估計"""
        if advi_trace is None:
            print(f"{model_name}: ADVI 失敗，無法比較")
            return
            
        print(f"\n{model_name}:")
        print("參數        MCMC均值    ADVI均值    差異")
        print("-" * 45)
        
        mcmc_summary = az.summary(mcmc_trace, var_names=param_list, kind="stats")
        advi_summary = az.summary(advi_trace, var_names=param_list, kind="stats")
        
        for param in mcmc_summary.index:
            if param in advi_summary.index:
                mcmc_mean = mcmc_summary.loc[param, "mean"]
                advi_mean = advi_summary.loc[param, "mean"]
                diff = abs(mcmc_mean - advi_mean)
                print(f"{param:>10}  {mcmc_mean:>8.3f}  {advi_mean:>8.3f}  {diff:>8.3f}")

    # 比較各模型的 MCMC 和 ADVI 結果
    compare_mcmc_advi(trace_full_MCMC, trace_full_ADVI, scalar_params, "完整 Clog-log 模型")
    compare_mcmc_advi(trace_fraction_MCMC, trace_fraction_ADVI, ["gamma", "beta", "theta"], "Fraction link 模型")

    # ========================= 7. 模型結構比較總結 =========================
    print("\n" + "="*60)
    print("模型結構比較總結")
    print("="*60)
    
    print("\n模型架構差異分析:")
    print("完整 Clog-log 模型:")
    print("  • 包含形狀參數 q（估計值）")
    print("  • 使用完整的 Clog-log 鏈接函數")
    print("  • 接受機率: P(a=1|s,x) = 1 - (c_ij/(c_ij + τ_ij*s_ij))^q")
    print("  • 參數數量: 7個（q, γ, 3個β, 2個θ）")
    print("  • 計算複雜度: 高")
    
    print("\nFraction link 模型:")
    print("  • 直接的比率形式鏈接函數")
    print("  • 接受機率: P(a=1|s,x) = s_ij / (δ_ij + κ_ij)")
    print("  • δ_ij = γ * exp(-X_ij^T * β), κ_ij = exp(x_ij^T * θ)")
    print("  • 參數數量: 6個（γ, 3個β, 2個θ）")
    print("  • 計算複雜度: 中等")

    
    print("\n視覺化結果總覽:")
    
    # GraphViz 模型結構圖
    if GRAPHVIZ_AVAILABLE:
        print("  ✓ 模型結構圖 (GraphViz):")
        print("    • model_structure_完整_Clog-log_模型.png")
        print("    • model_structure_Fraction_link_模型.png")
        print("    • 清楚顯示變數間的依賴關係")
    else:
        print("  ⚠️ GraphViz 不可用，跳過模型結構圖")
    
    # 後驗分佈視覺化
    if MATPLOTLIB_AVAILABLE:
        print("\n  ✓ 後驗分佈視覺化 (ArviZ + Matplotlib):")
        print("    • posterior_distributions_*.png (參數後驗分佈)")
        print("    • mcmc_traces_*.png (MCMC 軌跡圖)")
        print("    • prediction_comparison_*.png (預測vs觀測比較)")
        print("    • prior_posterior_comparison_*.png (先驗vs後驗)")
        print("    • energy_diagnostic_*.png (能量診斷)")
        print("    • 包含混淆矩陣、準確率分析等")
    else:
        print("  ⚠️ Matplotlib 不可用，跳過後驗分佈視覺化")
        print("    📝 安裝指令: pip install matplotlib")
    
    print("\n  視覺化功能說明:")
    print("    • 先驗預測採樣：驗證模型的先驗假設")
    print("    • 後驗分佈圖：顯示參數的不確定性")
    print("    • 軌跡圖：檢查 MCMC 收斂性")
    print("    • 預測比較：評估模型預測性能")
    print("    • 能量診斷：檢測採樣效率問題")
    
    
    print("\n 模型性能:")
    if accuracy_full is not None:
        print(f"- 完整 Clog-log 模型準確率: {accuracy_full:.3f}")
    if accuracy_fraction is not None:
        print(f"- Fraction link 模型準確率: {accuracy_fraction:.3f}")

    # ========================= 8. 詳細誤差比較分析 =========================
    print("\n" + "="*60)
    print("詳細誤差比較分析：完整模型 vs 簡化模型")
    print("="*60)
    
    def calculate_parameter_errors(trace, true_params, model_name):
        """計算參數估計誤差"""
        print(f"\n{model_name} 參數估計誤差分析:")
        print("-" * 50)
        
        # 獲取參數估計值
        summary = az.summary(trace, kind="stats")
        
        # 儲存誤差結果
        errors = {}
        
        # 檢查每個參數的誤差
        for param_name, true_value in true_params.items():
            if isinstance(true_value, (int, float)):
                # 標量參數
                if param_name in summary.index:
                    estimated_value = summary.loc[param_name, "mean"]
                    absolute_error = abs(true_value - estimated_value)
                    relative_error = absolute_error / abs(true_value) * 100 if true_value != 0 else float('inf')
                    
                    errors[param_name] = {
                        'true': true_value,
                        'estimated': estimated_value,
                        'absolute_error': absolute_error,
                        'relative_error': relative_error
                    }
                    
                    print(f"{param_name:>8}: 真實值={true_value:>8.3f}, 估計值={estimated_value:>8.3f}")
                    print(f"{'':>8}  絕對誤差={absolute_error:>8.3f}, 相對誤差={relative_error:>7.2f}%")
                else:
                    print(f"{param_name:>8}: 未在模型中估計")
                    
            elif isinstance(true_value, np.ndarray):
                # 向量參數
                param_errors = []
                print(f"{param_name} 向量參數:")
                
                for i, true_val in enumerate(true_value):
                    param_key = f"{param_name}[{i}]"
                    if param_key in summary.index:
                        estimated_val = summary.loc[param_key, "mean"]
                        absolute_error = abs(true_val - estimated_val)
                        relative_error = absolute_error / abs(true_val) * 100 if true_val != 0 else float('inf')
                        
                        param_errors.append({
                            'true': true_val,
                            'estimated': estimated_val,
                            'absolute_error': absolute_error,
                            'relative_error': relative_error
                        })
                        
                        print(f"  {param_key:>10}: 真實值={true_val:>8.3f}, 估計值={estimated_val:>8.3f}")
                        print(f"  {'':>10}  絕對誤差={absolute_error:>8.3f}, 相對誤差={relative_error:>7.2f}%")
                    else:
                        print(f"  {param_key:>10}: 未在模型中估計")
                
                errors[param_name] = param_errors
        
        return errors
    
    # 計算兩個模型的誤差
    errors_full = calculate_parameter_errors(trace_full_MCMC, true_params, "完整 Clog-log 模型")
    
    # Fraction link 模型的真實參數（不包含 q）
    true_params_fraction = {
        "gamma": true_params["gamma"],
        "beta": true_params["beta"],
        "theta": true_params["theta"]
    }
    errors_fraction = calculate_parameter_errors(trace_fraction_MCMC, true_params_fraction, "Fraction link 模型")
    
    # ========================= 誤差比較總結 =========================
    print("\n" + "="*60)
    print("誤差比較總結")
    print("="*60)
    
    print("\n共同參數誤差比較:")
    print("參數名稱        完整模型誤差    Fraction模型誤差    誤差差異")
    print("-" * 70)
    
    # 比較共同參數
    total_error_full = 0
    total_error_fraction = 0
    param_count = 0
    
    for param_name in ["gamma", "beta", "theta"]:
        if param_name in errors_full and param_name in errors_fraction:
            if isinstance(errors_full[param_name], dict):
                # 標量參數
                error_full = errors_full[param_name]['absolute_error']
                error_fraction = errors_fraction[param_name]['absolute_error']
                error_diff = error_full - error_fraction
                
                total_error_full += error_full
                total_error_fraction += error_fraction
                param_count += 1
                
                print(f"{param_name:>12}     {error_full:>8.3f}           {error_fraction:>8.3f}     {error_diff:>+8.3f}")
                
            elif isinstance(errors_full[param_name], list):
                # 向量參數
                for i, (err_full, err_fraction) in enumerate(zip(errors_full[param_name], errors_fraction[param_name])):
                    error_full = err_full['absolute_error']
                    error_fraction = err_fraction['absolute_error']
                    error_diff = error_full - error_fraction
                    
                    total_error_full += error_full
                    total_error_fraction += error_fraction
                    param_count += 1
                    
                    param_label = f"{param_name}[{i}]"
                    print(f"{param_label:>12}     {error_full:>8.3f}           {error_fraction:>8.3f}     {error_diff:>+8.3f}")
    
    print("-" * 70)
    avg_error_full = total_error_full / param_count if param_count > 0 else 0
    avg_error_fraction = total_error_fraction / param_count if param_count > 0 else 0
    
    print(f"{'平均誤差':>12}     {avg_error_full:>8.3f}           {avg_error_fraction:>8.3f}     {avg_error_full - avg_error_fraction:>+8.3f}")
    
    # ========================= 特有參數分析 =========================
    print(f"\n完整模型特有參數 (q):")
    if "q" in errors_full:
        q_error = errors_full["q"]
        print(f"  q 參數: 真實值={q_error['true']:.3f}, 估計值={q_error['estimated']:.3f}")
        print(f"         絕對誤差={q_error['absolute_error']:.3f}, 相對誤差={q_error['relative_error']:.2f}%")
    
    # ========================= 模型選擇建議 =========================
    print(f"\n🎯 模型選擇建議:")
    print("-" * 40)
    
    if avg_error_full < avg_error_fraction:
        print("✅ 完整 Clog-log 模型表現較佳")
        print(f"   • 平均絕對誤差更低: {avg_error_full:.3f} vs {avg_error_fraction:.3f}")
        print(f"   • 誤差改善: {((avg_error_fraction - avg_error_full) / avg_error_fraction * 100):.1f}%")
        better_model = "完整模型"
    elif avg_error_fraction < avg_error_full:
        print("✅ Fraction link 模型表現較佳")
        print(f"   • 平均絕對誤差更低: {avg_error_fraction:.3f} vs {avg_error_full:.3f}")
        print(f"   • 誤差改善: {((avg_error_full - avg_error_fraction) / avg_error_full * 100):.1f}%")
        better_model = "Fraction模型"
    else:
        print("⚖️ 兩個模型表現相當")
        print(f"   • 平均絕對誤差相近: {avg_error_full:.3f} ≈ {avg_error_fraction:.3f}")
        better_model = "相當"
    
  
    
    print("\n📈 實證結果總結:")
    print("-" * 50)
    
    # 總結模型性能
    if accuracy_full is not None and accuracy_fraction is not None:
        performance_diff = accuracy_full - accuracy_fraction
        print(f"預測準確率:")
        print(f"  • 完整 Clog-log 模型: {accuracy_full:.3f}")
        print(f"  • Fraction link 模型: {accuracy_fraction:.3f}")
        print(f"  • 準確率差異: {performance_diff:+.3f}")
        
        if abs(performance_diff) < 0.01:
            print(f"  → 兩種模型預測性能相當")
        elif performance_diff > 0:
            print(f"  → Clog-log 模型預測稍佳")
        else:
            print(f"  → Fraction link 模型預測稍佳")
    
    print(f"\n參數估計精度:")
    if 'avg_error_full' in locals() and 'avg_error_fraction' in locals():
        error_improvement = ((max(avg_error_full, avg_error_fraction) - min(avg_error_full, avg_error_fraction)) 
                            / max(avg_error_full, avg_error_fraction) * 100)
        
        if avg_error_full < avg_error_fraction:
            print(f"  • Clog-log 模型估計精度更高")
            print(f"  • 精度改善: {error_improvement:.1f}%")
        elif avg_error_fraction < avg_error_full:
            print(f"  • Fraction link 模型估計精度更高") 
            print(f"  • 精度改善: {error_improvement:.1f}%")
        else:
            print(f"  • 兩種模型估計精度相當")
    
    
