下面我幫你整理成「**最小且乾淨的實驗設計清單**」，專門用來評估 **DEFOCA 在 SSL（SimCLR / Barlow / VICReg / SwAV）中的效果**。
只列核心實驗組合，不展開細節。

---

# 一、Baseline 組（必要）

* ☐ SSL 原始設定（無 DEFOCA）
* ☐ SSL + 原本 GaussianBlur（如果原方法有）

---

# 二、DEFOCA 基本效果驗證

### 1️⃣ 插入位置固定：`global_aug → DEFOCA`

* ☐ + DEFOCA (contiguous)
* ☐ + DEFOCA (random)

---

# 三、Ratio Ablation（最關鍵）

固定 sigma，掃描：

* ☐ ratio = 0.1
* ☐ ratio = 0.2
* ☐ ratio = 0.3
* ☐ ratio = 0.4

觀察：

* linear probe
* kNN
* training loss stability

---

# 四、Sigma Ablation

固定 ratio（例如 0.2 或 0.3）：

* ☐ σ = 1
* ☐ σ = 2
* ☐ σ = 3

觀察 representation degradation vs robustness

---

# 五、Patch Selection Strategy 比較

* ☐ contiguous
* ☐ random
* ☐ dispersed（如果時間允許）

目標：驗證 spatial coherence 是否重要

---

# 六、與原 GaussianBlur 的交互作用

* ☐ 只用原 global blur
* ☐ 只用 DEFOCA
* ☐ global blur + DEFOCA
* ☐ global blur(prob 降低) + DEFOCA

測試是否 redundant 或 complementary

---

# 七、Multi-view 效果（進階）

如果方法是：

### SimCLR / Barlow / VICReg

* ☐ 2 views + DEFOCA
* ☐ 2 views（baseline）
* ☐ 4 views（multi-positive）+ DEFOCA（可選）

### SwAV

* ☐ DEFOCA only on global crops
* ☐ DEFOCA on global + local
* ☐ local weaker ratio

---

# 八、Training Stability 檢查（必要）

* ☐ Loss curve 是否震盪
* ☐ Representation std（VICReg 尤其重要）
* ☐ Collapse 指標（Barlow off-diagonal、VICReg var term）

---

# 九、Robustness 測試（加分）

在 validation 時加入：

* ☐ Gaussian noise
* ☐ global blur
* ☐ JPEG compression
* ☐ partial occlusion

測試 representation robustness 是否提升

---

# 十、Compute Overhead

* ☐ throughput (img/sec)
* ☐ GPU memory
* ☐ training time per epoch

證明 DEFOCA 是低成本 augmentation

---

# 最小可投稿版本（精簡推薦組合）

如果你想要「最小但有說服力」：

1. Baseline
2. +DEFOCA(contiguous, ratio=0.2)
3. ratio sweep (0.1/0.2/0.3)
4. strategy comparison (contiguous vs random)
5. robustness test

這樣就足夠構成一篇 augmentation enhancement 類型的實驗 section。

---

如果你願意，我可以再幫你：

* 🔬 設計一個「專門檢驗 representation frequency bias 改變」的分析實驗（會比較有 analysis paper 味道）
* 或幫你整理成「實驗章節結構模板」直接可寫進論文
