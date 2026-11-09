# Reuters 詞向量近鄰的實際觀察

CBOW 與 Skip-Gram 的固定十詞 Top-5 來自保存的 100 維原始向量；本頁是事後分析推論，沒有用這些觀察修改模型或查詢，也不是人工查詢相關性標註。

## 合理關係與新聞情境

- oil 的 Skip-Gram Top-5 為 crude、gas、shut-in、liquids、natural。這些詞可與能源生產或油氣新聞情境連結；鄰近並不表示它們全部都是 oil 的同義詞。
- petroleum 的 CBOW 近鄰包括 oil、energy、esso、columbia、gas；其中能源詞及公司名稱可能共享新聞上下文。公司名稱的接近不能直接解讀成可交換的普通名詞。
- profit 的第一近鄰在兩模型都是 loss，cosine 分別為 0.655422 與 0.739965。訓練集有 569 篇同時含有這兩詞；例如 NEWID 13 的財報同時對照損失與獲利。相近的財報句式可解釋上下文接近，但它們在方向上相反，不能當成同義詞。

## 看似意外的結果及可能原因

- oil 的 CBOW 第一近鄰是 kernel。kernel 在訓練集 CF=10、DF=6，與 oil 共現於 5 篇新聞；NEWID 885 涉及棕櫚油及棕櫚仁的出口稅。oil 的多義性及少數相似情境可能使它靠近 kernel，不能直接判定為錯誤或原油同義。
- profit 近鄰包含 688、373.3、2.68 等數值 token。688 的 CF=15、DF=15；NEWID 1191 中有 4,688,000，逗號分詞會留下 688。財報模板、數字切詞及低頻分布是合理的可能原因，並非語意證明。
- oil 的 CBOW 近鄰 inter-city 在訓練集僅 CF=2、DF=2；與 oil 共現 1 篇。低頻詞的鄰近可能不穩定，需更多語料或不同 seed 才能評估穩定性。
- petroleum 的 Skip-Gram 近鄰 wpla 的 CF=2、DF=2，兩篇均與 petroleum 共現；NEWID 11168 的公司標題含 WOODSIDE PETROLEUM 及交易代號 WPLA.S。這反映具體公司的上下文，不能當成 petroleum 的普遍同義詞。
- trade 的 CBOW 近鄰 invisibles 的 CF=11、DF=7，這七篇均與 trade 共現，NEWID 1890 包含相關貿易收支數字。這是新聞情境關係的線索，未逐篇判定詞義。

## 文件檢索與詞近鄰必須分開解讀

十查詢平均：TF-IDF 的 P@10=0.86、MRR=1.00、nDCG@10=0.8840；加權 CBOW 為 0.90、0.90、0.8828；加權 Skip-Gram 為 0.93、0.95、0.9266。Skip-Gram 在本次 P@10 與 nDCG@10 較高，但 MRR 仍低於 TF-IDF，不能從詞近鄰或單一指標宣稱普遍優勢。完整查詢與排名見 evaluation-v1/REPORT.md。

## 完整固定十詞的 Top-5

沒有事後移除數字、縮寫或看似不合理的詞，兩模型均搜尋各自完整 23,172 詞詞表。

### profit

| 模型 | 名次 | 鄰近詞 | cosine |
|---|---:|---|---:|
| cbow | 1 | loss | 0.655422 |
| cbow | 2 | 688 | 0.616559 |
| cbow | 3 | 373.3 | 0.585733 |
| cbow | 4 | vs | 0.581894 |
| cbow | 5 | profits | 0.548839 |
| skipgram | 1 | loss | 0.739965 |
| skipgram | 2 | net | 0.675822 |
| skipgram | 3 | 2.68 | 0.667478 |
| skipgram | 4 | 688 | 0.667040 |
| skipgram | 5 | 108.3 | 0.663702 |

### loss

| 模型 | 名次 | 鄰近詞 | cosine |
|---|---:|---|---:|
| cbow | 1 | oper | 0.726014 |
| cbow | 2 | vs | 0.682799 |
| cbow | 3 | 571 | 0.665728 |
| cbow | 4 | 1.40 | 0.659956 |
| cbow | 5 | profit | 0.655422 |
| skipgram | 1 | net | 0.779277 |
| skipgram | 2 | oper | 0.771615 |
| skipgram | 3 | 5.17 | 0.741170 |
| skipgram | 4 | profit | 0.739965 |
| skipgram | 5 | 31.09 | 0.732442 |

### earnings

| 模型 | 名次 | 鄰近詞 | cosine |
|---|---:|---|---:|
| cbow | 1 | profits | 0.659685 |
| cbow | 2 | income | 0.631860 |
| cbow | 3 | revenues | 0.584460 |
| cbow | 4 | results | 0.540163 |
| cbow | 5 | losses | 0.480831 |
| skipgram | 1 | 65.0 | 0.682235 |
| skipgram | 2 | minorco | 0.650297 |
| skipgram | 3 | telxon | 0.650070 |
| skipgram | 4 | full-year | 0.649448 |
| skipgram | 5 | thrid | 0.646374 |

### acquisition

| 模型 | 名次 | 鄰近詞 | cosine |
|---|---:|---|---:|
| cbow | 1 | merger | 0.691048 |
| cbow | 2 | acquisitions | 0.621937 |
| cbow | 3 | transaction | 0.613391 |
| cbow | 4 | purchase | 0.579135 |
| cbow | 5 | buyout | 0.567054 |
| skipgram | 1 | transaction | 0.644007 |
| skipgram | 2 | purchase | 0.638126 |
| skipgram | 3 | merger | 0.624891 |
| skipgram | 4 | buyout | 0.617325 |
| skipgram | 5 | sahlen | 0.614532 |

### merger

| 模型 | 名次 | 鄰近詞 | cosine |
|---|---:|---|---:|
| cbow | 1 | acquisition | 0.691048 |
| cbow | 2 | transaction | 0.571210 |
| cbow | 3 | buyout | 0.546119 |
| cbow | 4 | kistler | 0.538669 |
| cbow | 5 | reorganization | 0.534508 |
| skipgram | 1 | str | 0.658696 |
| skipgram | 2 | rni | 0.647238 |
| skipgram | 3 | vons | 0.629724 |
| skipgram | 4 | rainbow's | 0.625886 |
| skipgram | 5 | acquisition | 0.624891 |

### oil

| 模型 | 名次 | 鄰近詞 | cosine |
|---|---:|---|---:|
| cbow | 1 | kernel | 0.534936 |
| cbow | 2 | natural | 0.518681 |
| cbow | 3 | petroleum | 0.500589 |
| cbow | 4 | inter-city | 0.483376 |
| cbow | 5 | crude | 0.452701 |
| skipgram | 1 | crude | 0.777472 |
| skipgram | 2 | gas | 0.691155 |
| skipgram | 3 | shut-in | 0.634981 |
| skipgram | 4 | liquids | 0.618474 |
| skipgram | 5 | natural | 0.609948 |

### petroleum

| 模型 | 名次 | 鄰近詞 | cosine |
|---|---:|---|---:|
| cbow | 1 | oil | 0.500589 |
| cbow | 2 | energy | 0.490081 |
| cbow | 3 | esso | 0.484941 |
| cbow | 4 | columbia | 0.464873 |
| cbow | 5 | gas | 0.456656 |
| skipgram | 1 | oil | 0.596424 |
| skipgram | 2 | wpla | 0.579874 |
| skipgram | 3 | gas | 0.571151 |
| skipgram | 4 | clnp | 0.563075 |
| skipgram | 5 | norte | 0.562987 |

### wheat

| 模型 | 名次 | 鄰近詞 | cosine |
|---|---:|---|---:|
| cbow | 1 | barley | 0.651183 |
| cbow | 2 | corn | 0.630611 |
| cbow | 3 | soybeans | 0.582409 |
| cbow | 4 | grains | 0.572603 |
| cbow | 5 | grain | 0.570893 |
| skipgram | 1 | barley | 0.736833 |
| skipgram | 2 | durum | 0.710800 |
| skipgram | 3 | winter | 0.679617 |
| skipgram | 4 | flour | 0.674438 |
| skipgram | 5 | grain | 0.669013 |

### grain

| 模型 | 名次 | 鄰近詞 | cosine |
|---|---:|---|---:|
| cbow | 1 | grains | 0.620773 |
| cbow | 2 | corn | 0.606883 |
| cbow | 3 | wheat | 0.570893 |
| cbow | 4 | soybeans | 0.532448 |
| cbow | 5 | sugar | 0.523823 |
| skipgram | 1 | coarse | 0.678720 |
| skipgram | 2 | wheat | 0.669013 |
| skipgram | 3 | grains | 0.666524 |
| skipgram | 4 | corn | 0.652059 |
| skipgram | 5 | crop | 0.650804 |

### trade

| 模型 | 名次 | 鄰近詞 | cosine |
|---|---:|---|---:|
| cbow | 1 | warn | 0.467030 |
| cbow | 2 | invisibles | 0.457420 |
| cbow | 3 | bilateral | 0.451902 |
| cbow | 4 | retaliation | 0.442804 |
| cbow | 5 | gatt | 0.441432 |
| skipgram | 1 | full-scale | 0.627403 |
| skipgram | 2 | surplus | 0.588324 |
| skipgram | 3 | aggravation | 0.586381 |
| skipgram | 4 | tensions | 0.585891 |
| skipgram | 5 | deplore | 0.581505 |
