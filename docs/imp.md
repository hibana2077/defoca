下面是一份針對你這份稿件中的 **DEFOCA（Blur-to-Focus Attention Layer）核心演算法** 的「實現指引」（偏工程落地、可直接轉成 PyTorch / torchvision 實作）。內容依照文中定義：**先做全域 augmentation，再做 patchify → 選 patch → 對選到的 patch 做 Gaussian blur（low-pass）→ 生成多視角 V views；測試時移除 DEFOCA**。

---

## 1) 模組定位與介面設計

**定位**：DEFOCA 是 *training-time* 的資料轉換層（data transform / augmentation layer），不改 backbone 架構；在訓練時 on-the-fly 產生多個視角，推論時不使用。

**建議介面**

* `forward(x) -> views`

  * `x`: 單張 `(C,H,W)` 或 batch `(B,C,H,W)`
  * `views`: 單視角回傳同形狀；多視角回傳 `(V,C,H,W)` 或 `(B,V,C,H,W)`（看你 pipeline 怎麼接）

**必要超參數**

* `P`: 每個維度切成 P 格，總 patch 數 `N = P^2` 
* `ratio`: 選取比例 `n/N`（或直接給 `n`）
* `sigma`: Gaussian blur 強度（σ）
* `strategy`: `random | contiguous | dispersed`（文中對比用）
* `V`: 視角數（文中示例 `V=8`） 
* `max_attempts`: 多視角時為了避免 index set 重複，允許重抽的次數（Appendix 有描述）

---

## 2) Step A：Patchification（切格子 + 索引映射）

把影像切成 `P×P` 的非重疊網格，共 `N=P^2` 個 patch。 

### 2.1 邊界處理（很重要）

當 `H` 或 `W` 不能被 `P` 整除時：

* 先用 `pw = floor(W/P)`, `ph = floor(H/P)`
* **最後一列 / 最後一行**吸收剩餘像素，以確保完整覆蓋（exact tiling） 

### 2.2 Patch index 定義（建議）

以 `(r, c)` 表示第 r 列第 c 行（0-based），線性 index：

* `idx = r * P + c`
* 反查：`r = idx // P`, `c = idx % P`

並提供 `bbox(idx)`：

* `x0 = c*pw`, `x1 = (c+1)*pw`，但若 `c==P-1` 則 `x1=W`
* `y0 = r*ph`, `y1 = (r+1)*ph`，但若 `r==P-1` 則 `y1=H`

---

## 3) Step B：Patch Selection（選 n 個要 blur 的 patch）

令 `n = round(ratio * N)`（或直接指定）。文中三種策略：

### 3.1 Random（O(n)）

* 從 `[0..N-1]` **不重複**抽 n 個 index。 

### 3.2 Contiguous（推薦，O(n)）

**做法（Appendix A.5 的工程版敘述）**：
從隨機 seed patch 開始，往上下左右鄰居擴張，直到收集到 n 個；若擴張卡住，剩餘不足的部分 fallback 回 random 補齊。 

鄰居定義（4-neighborhood）：

* `(r-1,c) (r+1,c) (r,c-1) (r,c+1)`（要做邊界檢查）

### 3.3 Dispersed（覆蓋高但較慢，Θ(nN)）

* 使用 farthest-point heuristic：每一步挑一個 patch，使它到「已選集合」的最小平方距離最大。
  （文中也指出這策略 selection cost 會顯著偏高）

---

## 4) Step C：Patch Operation（對選到的 patch 做 low-pass defocus）

核心操作：對選到的 patch 套 **Gaussian blur**（low-pass），未選到的 patch 保持不變。

> 文中用 Fourier 形式寫成 `p'_i = F^{-1}(H_{σ}(ω) · F(p_i)(ω))`，本質就是抑制高頻、保留低頻結構。 

### 4.1 工程實作選項

你可以用任一種方式做「對 patch 做 blur」：

**(A) 直接對 patch tensor 做 GaussianBlur（最直觀）**

* 取出 patch：`patch = x[:, y0:y1, x0:x1]`
* 對 patch 執行 blur（σ 固定或每個 patch 可變）
* 把結果貼回原圖（crop-process-paste） 

**(B) 自己用 separable conv 寫 Gaussian blur（較可控）**

* kernel size 常用：`k = 2*ceil(3σ)+1`
* 先 1D 橫向再 1D 縱向（separable），`groups=C` 做 depthwise
* patch 內 padding 建議用 `reflect` / `replicate`，避免引入黑邊（同時避免跨 patch 讀到外部像素）

> Appendix 也提到可用 separable convolutions 等方式做硬體友善優化。 

---

## 5) Step D：Multi-view 生成（V 個不同 patch layout）

文中強調：同一張輸入影像，重複抽不同 patch set，得到 `V` 個 views（例：`V=8`），形成 soft attention/regularization 效果。

**實作重點**

1. 迴圈 `v=1..V`：

   * 抽一組 index set `I_v`
   * 套用 Step C 得到 `x_v`
2. **避免重複**（可選但建議）：

   * 用 `hashset` 記錄 `tuple(sorted(I_v))`
   * 若撞到就重抽，最多 `max_attempts` 次
3. 最後輸出：

   * `stack(views)` → `(V,C,H,W)`（或 batch 情況 `(B,V,C,H,W)`）

---

## 6) Training pipeline 串接方式（務必做對）

### 6.1 插入位置

依文中描述：**先做全域 augmentation（crop/flip/color jitter…），再做 DEFOCA**。

### 6.2 Loss 怎麼算（兩種常見做法）

假設原本 batch 是 `(B,C,H,W)`、label 是 `(B,)`：

**做法 1：把 view 維度展平成 batch**

* 產生 `(B,V,C,H,W)` → reshape 成 `(B*V,C,H,W)`
* label 重複 V 次：`y.repeat_interleave(V)`
* loss 正常算（等價於把每個 view 當獨立樣本）

**做法 2：同一張圖的 V views 做平均 loss**

* 對每個樣本 i：`loss_i = mean_v CE(model(x_{i,v}), y_i)`
* 再對 batch 平均

> Appendix 也明確指出 multi-view 會讓每個 epoch 的計算近似隨 V 線性成長，VRAM 也約 ∝ B×V（所以常要 trade-off batch size vs V）。

### 6.3 推論（Testing）

**關掉 DEFOCA**（直接走原本乾淨的 inference pipeline）。 

---

## 7) 參考偽代碼（可直接翻成 PyTorch）

```python
def defoca_single(x, P, n, sigma, strategy, rng):
    # x: (C,H,W)
    C, H, W = x.shape
    pw, ph = W // P, H // P

    def bbox(idx):
        r, c = idx // P, idx % P
        x0, x1 = c * pw, (c + 1) * pw
        y0, y1 = r * ph, (r + 1) * ph
        if c == P - 1: x1 = W
        if r == P - 1: y1 = H
        return y0, y1, x0, x1

    I = select_indices(P, n, strategy, rng)  # random / contiguous / dispersed
    out = x.clone()

    for idx in I:
        y0, y1, x0, x1 = bbox(idx)
        patch = out[:, y0:y1, x0:x1]
        patch_blur = gaussian_blur(patch, sigma)  # 任一實作
        out[:, y0:y1, x0:x1] = patch_blur

    return out


def multiview_defoca(x, P, ratio, sigma, strategy, V, max_attempts, rng):
    N = P * P
    n = int(round(ratio * N))
    views = []
    used = set()

    for _ in range(V):
        for _attempt in range(max_attempts):
            I = select_indices(P, n, strategy, rng)
            key = tuple(sorted(I))
            if key not in used:
                used.add(key)
                break
        views.append(defoca_single(x, P, n, sigma, strategy, rng))

    return stack(views, dim=0)  # (V,C,H,W)
```

---

## 8) 實作檢查清單（避免常見踩雷）

1. **確保只 blur 被選到的 patch**：未選 patch 必須 bitwise/數值一致（方便單元測試）。
2. **邊界 patch 尺寸正確**：最後一列/行吸收剩餘像素，避免漏覆蓋或越界。 
3. **Contiguous expansion 可能卡住**：務必做 fallback random 補齊（Appendix 明講）。 
4. **多視角唯一性**：若你要求 unique sets，記得 `max_attempts`，並接受極端情況可能會撞到（Appendix 有討論）。 
5. **訓練/推論模式**：eval 時不要套 DEFOCA。 

---

如果你接下來希望我把它「落成可跑的 PyTorch 模組（含 contiguous/dispersed 選取、tensor/PIL 兩種版本、batch+multiview reshape、seed 可重現）」我也可以直接給一份乾淨的 reference implementation。
