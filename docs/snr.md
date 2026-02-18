很好，這題我們直接用論文 Section 2.3 的定義來算（Eq.8），然後我會給你一個**可實作版本（PyTorch 風格）**。

---

# 一、理論定義（來自 Eq.8）

論文定義：

[
\text{SNR} =
\frac{
\sum_{i \in S} |F(p_i)|^2
}{
\sum_{i \notin S} |F(p_i)|^2
}
]

其中：

* ( S ) = discriminative patches index set
* ( p_i ) = 第 i 個 patch
* ( F(p_i) ) = Fourier transform
* ( |F(p_i)|^2 ) = 頻域能量（power spectrum）

DEFOCA 後：

[
\text{SNR}' =
\frac{
\sum_{i \in S} |F(p_i)|^2
}{
\sum_{i \notin S} |H_\sigma(\omega)F(p_i)|^2
}
]

因為 ( 0 \le H_\sigma(\omega) \le 1 )，理論上：

[
\text{SNR}' > \text{SNR}
]

---

# 二、實際怎麼算？（工程可跑版本）

## Step 1️⃣：取得 discriminative set S

你需要一個 proxy：

### 方法 A（推薦）

用：

* Grad-CAM
* attention map（ViT last-layer）
* saliency map

然後：

1. resize 到 patch grid
2. 每個 patch 計算平均 saliency
3. top-k% 當作 S

---

## Step 2️⃣：對每個 patch 做 FFT

對 patch ( p_i \in \mathbb{R}^{C \times H_p \times W_p} )

用 2D FFT：

```python
import torch

def patch_energy(patch):
    # patch: (C, H, W)
    fft = torch.fft.fft2(patch, norm="ortho")
    power = torch.abs(fft) ** 2
    return power.sum()
```

這個 `.sum()` 就是：

[
|F(p_i)|^2
]

---

## Step 3️⃣：分開累加

```python
def compute_snr(patches, S):
    signal_energy = 0.0
    noise_energy = 0.0
    
    for i, p in enumerate(patches):
        e = patch_energy(p)
        if i in S:
            signal_energy += e
        else:
            noise_energy += e
            
    return signal_energy / (noise_energy + 1e-8)
```

---

# 三、如果你要算「高頻能量」版本（更精準）

論文本質是 suppress high-frequency。

更精準版本應該：

[
\text{HighFreqEnergy} =
\sum_{|\omega| > \omega_c}
|F(p_i)(\omega)|^2
]

實作：

```python
def high_freq_energy(patch, cutoff_ratio=0.5):
    fft = torch.fft.fft2(patch, norm="ortho")
    power = torch.abs(fft) ** 2
    
    H, W = power.shape[-2:]
    cy, cx = H//2, W//2
    
    yy, xx = torch.meshgrid(
        torch.arange(H), torch.arange(W), indexing="ij"
    )
    dist = torch.sqrt((yy-cy)**2 + (xx-cx)**2)
    
    cutoff = cutoff_ratio * max(H, W)
    mask = dist > cutoff
    
    return power[..., mask].sum()
```

然後：

```
SNR = signal_total_energy / noise_highfreq_energy
```

這版本更貼近「DEFOCA 提升 discriminative SNR」的理論精神。

---

# 四、實驗流程（完整 protocol）

對每張圖：

1. 切成 P×P patches

2. 用 attention 找 S

3. 計算：

   * Baseline SNR
   * DEFOCA SNR'

4. 統計：

   * mean
   * std
   * paired t-test

---

# 五、你應該畫的圖

### 📈 建議 1

Histogram:

* Baseline SNR
* DEFOCA SNR'

應該右移。

---

### 📊 建議 2

Scatter plot：

x-axis: Baseline SNR
y-axis: DEFOCA SNR

若大部分點在 y=x 上方 → 理論成立。

---

# 六、重要細節（reviewer 會問的）

### 1️⃣ 為什麼 discriminative 用 saliency？

你要說：

> S is approximated by model-derived attention proxy.

---

### 2️⃣ 是否 circular reasoning？

避免用：

* DEFOCA 訓練出來的 attention 來定義 S
* 再說 DEFOCA 提升 SNR

最好用：

* Baseline model 來定義 S
* 再測 DEFOCA

---

### 3️⃣ 用哪個頻率範圍？

如果 reviewer 很刁鑽：

你可以：

* 分成 low / mid / high frequency band
* show DEFOCA 主要 suppress high-frequency noise

---

# 七、如果你想讓這變成 Analysis Paper 級別

可以再加：

[
\Delta SNR = \text{SNR}' - \text{SNR}
]

然後檢查：

* 與 accuracy 提升的 correlation
* 與 Psafe 的 correlation
* 與 representation drift 的 correlation

如果三者線性相關，你這篇 paper 會很強。

---

# 總結

實際計算就是：

[
\boxed{
\text{SNR} =
\frac{\sum_{i \in S} \sum_\omega |F(p_i)(\omega)|^2}
{\sum_{i \notin S} \sum_\omega |F(p_i)(\omega)|^2}
}
]

然後比較：

[
\text{SNR}' > \text{SNR}
]

---

如果你告訴我：

* 你用 CNN 還是 ViT
* patch size
* dataset

我可以幫你設計一個「最穩定、不會被 reviewer 攻擊」的 SNR 計算 protocol。
