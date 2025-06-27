
### 套件清單
已安裝版本：
- pymc: 5.22.0
- arviz: 0.21.0  
- numpy: 1.26.4
- pandas: 2.2.3
- matplotlib: 3.10.0
- polars: 1.29.0
- pytensor: 2.30.3
- graphviz: 0.21
- jupyter: 1.1.1
- 
### 一鍵安裝（Conda）
```bash
conda install -c conda-forge pymc arviz numpy pandas matplotlib polars graphviz jupyter
```

### 1. PyMC_BuyerSeller_ClogLog_250527.py
**買賣雙方互動接受行為模型 - 鏈接函數比較研究**

- **主要功能**：比較兩種鏈接函數建模賣家接受決策的效果
- **核心技術**：
  - **Fraction Link**：P(a=1|s,x) = s_ij / (δ_ij + κ_ij)
  - **Complementary Log-log Link**：P(a=1|s,x) = 1 - (c_ij/(c_ij + τ_ij*s_ij))^q
- **模擬資料**：生成賣家回應時間(SRT)與接受行為(SA)資料
- **推論方法**：NUTS MCMC 與 ADVI 變分推論
- **執行環境**：Windows PowerShell
- **執行指令**：
  ```powershell
  python PyMC_BuyerSeller_ClogLog_250527.py
  ```

### 2. PyMC_Reg_250527_no_y2.py
**三方程式系統與相關隨機效應模型**

- **主要功能**：建構與估計具有相關隨機效應的三方程式系統
- **模型特色**：
  - **方程式結構**：y1 → y3, y4（y1影響y3和y4）
  - **分佈類型**：y1,y3為Gamma分佈，y4為Bernoulli分佈
  - **隨機效應**：個體特定的相關隨機效應u
- **模型比較**：完整模型 vs 忽略隨機效應的簡化模型
- **推論方法**：NUTS、MB-ADVI與ADVI變分推論
- **修正內容**：移除y2變數，優化Windows多進程支援
- **執行指令**：
  ```powershell
  python PyMC_Reg_250527_no_y2.py
  ```

### 3. clean_data_final.ipynb
**資料清理與前處理筆記本**
- **主要功能**：原始資料的清理、轉換與預處理
- **處理流程**：資料品質檢查、異常值處理、特徵工程
- **輸出格式**：為後續建模分析準備的乾淨資料集
