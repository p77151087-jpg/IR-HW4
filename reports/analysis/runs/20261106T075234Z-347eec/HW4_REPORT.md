# HW4 Reuters 詞彙與文件檢索完整報告

本實驗比較 TF-IDF、TF-IDF 加權 CBOW 與 TF-IDF 加權 Skip-Gram。詞近鄰與文件排名分開檢驗：前者說明上下文關係，後者以固定查詢與 Reuters topics 衡量排序。

本次十查詢平均 P@10 以 Skip-Gram 較高，但 TF-IDF 的 MRR 較高；CBOW 的 nDCG@10 也略低於 TF-IDF。結果不支持所有 embedding 方法在所有指標上均改善的結論。

## 一 資料與實驗控制

Reuters-21578 Distribution 1.0，ModApte：9,603 篇訓練、3,299 篇測試、8,676 篇 unused。
按 TOPICS=YES 與 LEWISSPLIT=TRAIN/TEST 定義切分，保留空 TOPICS 元素、多標籤與其他主題的測試新聞。五主題 earn、acq、crude、grain、trade 各設兩個查詢；排名候選是完整測試集。
只以訓練集建立詞表、DF、IDF 及 Word2Vec。標題與正文參與計分，日期、ID、topics 不進入向量。NFC、casefold、撇號／連字號正規化、固定 132 停用詞、不 stemming；保留 no、not、without。
訓練序列為 70,334 句、888,403 tokens，詞表 39,012 詞。句子邊界不跨文件或標題／正文；前處理後逐篇 token 與完整文字核對相同。

## 二 方法與訓練流程

TF 是單篇新聞或查詢的出現次數；CF 是訓練集合總次數；DF 是含詞的訓練新聞篇數。IDF(t)=log10(N_train/DF_train(t))，TF-IDF(t,d)=TF(t,d)×IDF(t)。
新聞與查詢使用同一詞表、IDF 與實際重複 TF。TF-IDF cosine 是兩向量點積除以完整詞彙空間的 L2 norm。
加權詞向量 v(d)=Σ[TF-IDF(t,d)×e(t)]／Σ[TF-IDF(t,d)]，求和限於有 Word2Vec 向量的詞。查詢也用相同公式，再計算原始 100 維 cosine；OOV 忽略並回報涵蓋率。
CBOW 以周邊上下文預測中心詞，Skip-Gram 以中心詞預測周邊上下文。例如示意序列 crude oil prices：CBOW 可由 crude／prices 預測 oil，Skip-Gram 可由 oil 預測 crude／prices。這只是流程示意，不是新增訓練語料。
兩模型均由新 Gensim Word2Vec constructor、build_vocab、train 從頭訓練；設定 negative sampling、記錄逐 epoch loss，保存完整模型並驗證重載相同。Gensim 執行訓練核心，未宣稱手寫優化器。

| 參數 | CBOW | Skip-Gram |
|---|---:|---:|
| sg | 0 | 1 |
| vector_size | 100 | 100 |
| window | 5 | 5 |
| min_count | 2 | 2 |
| epochs | 30 | 30 |
| seed | 42 | 42 |
| workers | 1 | 1 |
| negative | 5 | 5 |
| sample | 0.001 | 0.001 |
| alpha | 0.025 | 0.025 |
| min_alpha | 0.0001 | 0.0001 |

其餘設定相同，包括 ns_exponent=0.75、cbow_mean=1、sorted_vocab=1、shrink_windows=True。每模型 23,172 詞；未使用預訓練或舊 HW3 權重。單 seed 與單組參數未經最佳化比較，loss 不是語意正確率。
所有有效測試向量參與排序，embedding 沒有共同查詢詞門檻。零分及負 cosine 保留，零向量排除並計數；同分先按 12 位小數分數再按數字 NEWID 升冪，保存原始精度。

## 三 固定查詢與相關性

| 查詢 | 文字 | 目標 topic | 意圖 |
|---|---|---|---|
| Q01 | corporate profit earnings | earn | 企業獲利與盈餘 |
| Q02 | quarterly net income | earn | 季度淨收益 |
| Q03 | company acquisition merger | acq | 企業收購與合併 |
| Q04 | takeover bid shares | acq | 收購要約與股權 |
| Q05 | crude oil production | crude | 原油生產 |
| Q06 | petroleum output prices | crude | 石油產出與價格 |
| Q07 | grain wheat corn | grain | 穀物、小麥與玉米 |
| Q08 | wheat grain exports | grain | 小麥與穀物出口 |
| Q09 | international trade deficit | trade | 國際貿易逆差 |
| Q10 | import export trade | trade | 進出口貿易 |

相關 iff 查詢的目標 topic 存在於原始文件 topics。這是資料集主題代理，不是針對查詢的人工判讀；分析推論不會改寫標籤或分數。

## 四 十個詞的 Top-5 與逐詞解釋

本次固定模型的分析推論，非逐篇人工相關性標註。數據與原文另列為可核對證據；不以文件共現直接證明詞義或訓練窗口內共現，不修改查詢、topics 或既有排名。

表格逐項說明近鄰的關係與新聞用法。標記「意外」的列包含低頻數字、專名或容易跨主題解讀的結果。共現是兩詞同時出現的訓練新聞篇數，後面的新聞 ID 可回查原文。

網站關係圖使用原始 100 維 cosine。各線長是中心詞與近鄰的單位向量距離 √(2 − 2 × cosine)。兩圖同尺度，連線方向由排版決定。所有近鄰的詞頻、共現與原文片段也收錄在證據包的 analysis.json。

### profit

兩模型共有 2 個 Top-5 鄰近詞（loss、688）。

#### CBOW

| 名次 | 鄰近詞 | cosine | 關係與解讀 | 共現篇數／訓練新聞 ID |
|---:|---|---:|---|---|
| 1 | loss | 0.655422 | **反向概念**。財報用同一欄位對照本期虧損和前期獲利。兩詞能出現在同樣的句型，盈虧方向仍然相反。 | 569／13 |
| 2 | 688 | 0.616559 | **金額切詞片段（意外）**。RAYTECH 財報的淨利金額 4,688,000 被逗號切成數字片段。少數金額用例可能讓這個片段接近 profit。 | 5／1191 |
| 3 | 373.3 | 0.585733 | **低頻財報金額（意外）**。金礦公司財報用這個數字列稅前與淨額。兩詞在訓練集中沒有同篇紀錄，可比較各自的財務用法。 | 0／8、2279 |
| 4 | vs | 0.581894 | **比較標記**。季度表用 vs 對照兩期數值，獲利與虧損都能出現在比較欄。 | 565／13 |
| 5 | profits | 0.548839 | **同詞單複數**。COMALCO 的標題寫 profits，內文用 profit 描述同一個轉虧為盈的事件。 | 102／921 |

#### Skip-Gram

| 名次 | 鄰近詞 | cosine | 關係與解讀 | 共現篇數／訓練新聞 ID |
|---:|---|---:|---|---|
| 1 | loss | 0.739965 | **反向概念**。財報用同一欄位對照本期虧損和前期獲利。兩詞能出現在同樣的句型，盈虧方向仍然相反。 | 569／13 |
| 2 | net | 0.675822 | **財報搭配詞**。net profit 指淨利。net 也修飾 net loss，提供的是淨額概念。 | 646／13 |
| 3 | 2.68 | 0.667478 | **低頻營收數字（意外）**。有一篇財報在盈虧列後報出 2.68 billion 的營收。數字的財報位置可能解釋接近。 | 2／4797 |
| 4 | 688 | 0.667040 | **金額切詞片段（意外）**。RAYTECH 財報的淨利金額 4,688,000 被逗號切成數字片段。少數金額用例可能讓這個片段接近 profit。 | 5／1191 |
| 5 | 108.3 | 0.663702 | **低頻獲利數字（意外）**。HUDSON'S BAY 用 108.3 mln 報前一期獲利。模型可能捕捉到數值和 profit 的報表搭配。 | 3／7883 |

**合理關係**　CBOW 的 profits 是同詞的屈折形式；Skip-Gram 的 net 與淨利語境有關。loss 在兩模型都很接近，但它與 profit 的獲利方向相反，不能當作同義詞。

**新聞情境**　財報常以 profit／loss、net、vs 與數字對照不同期間。這些詞處在相似表格及句式的位置，可能因此學到相近表示；下面的 profit／loss 共現新聞可支持這種情境解讀。

**意外結果與可能原因**　688、373.3、2.68、108.3 等數值出現在 Top-5，顯示財報模板、低頻數字與切詞方式可能影響近鄰。688 也會由 4,688,000 的逗號分割產生；不能將這些數字解讀為 profit 的語義同類。

**原文證據示例**

| 證據詞 | 訓練 CF | 訓練 DF | 與分析詞共現篇數 |
|---|---:|---:|---:|
| loss | 4046 | 1100 | 569 |
| 688 | 15 | 15 | 5 |

證據詞：loss

訓練新聞 13：AM INTERNATIONAL INC <AM> 2ND QTR JAN 31

> AM INTERNATIONAL INC <AM> 2ND QTR JAN 31 Oper shr loss two cts vs profit seven cts     Oper shr profit 442,000 vs profit 2,986,000     Revs 291.8 mln vs 151.1 mln     Avg shrs 51.7 mln vs 43.4 mln     Six mths     Oper shr profit nil vs profit 12 cts  

證據詞：688

訓練新聞 1191：RAYTECH CORP <RAY> 4TH QTR DEC 28 NET

> RAYTECH CORP <RAY> 4TH QTR DEC 28 NET Shr profit 78 cts vs loss 1.05 dlrs     Net profit 2,336,000 vs loss 3,002,000     Revs 26.0 mln vs 26.7 mln     Year     Shr profit 1.59 dlrs vs loss 6.35 dlrs     Net profit 4,688,000 vs loss 18.2 mln     Revs 113.5 mln vs 112.4 mln     NOTE: 1986 4th qtr and yr includes loss carryfoward of 534,000 dlrs and 1,662,000 dlrs, respectively.  Reuter

此處是文件共現，不是窗口內共現；共現為零時列出分開的原文情境，不能宣稱有直接共現。

### loss

兩模型共有 2 個 Top-5 鄰近詞（oper、profit）。

#### CBOW

| 名次 | 鄰近詞 | cosine | 關係與解讀 | 共現篇數／訓練新聞 ID |
|---:|---|---:|---|---|
| 1 | oper | 0.726014 | **營運損益欄**。財報用 Oper shr loss 和 Oper net loss 報營運損失。oper 標示營運口徑。 | 189／13 |
| 2 | vs | 0.682799 | **比較標記**。虧損金額後常接 vs 和另一期間的盈虧數字，形成固定比較句式。 | 761／13 |
| 3 | 571 | 0.665728 | **金額切詞片段（意外）**。BASIX 財報的 7,571,000 是比較期間停業損失金額，逗號使 571 成為獨立詞。 | 11／837 |
| 4 | 1.40 | 0.659956 | **低頻每股數字（意外）**。同篇報導一處談稅務虧損，另一處談每股價值估計 1.40 美元。這是同篇的財務情境，兩處沒有緊鄰。 | 5／6202 |
| 5 | profit | 0.655422 | **反向概念**。同一條每股財報可對照一季 loss 與另一季 profit。向量靠近反映共享句型，兩詞的盈虧方向相反。 | 569／13 |

#### Skip-Gram

| 名次 | 鄰近詞 | cosine | 關係與解讀 | 共現篇數／訓練新聞 ID |
|---:|---|---:|---|---|
| 1 | net | 0.779277 | **財報搭配詞**。net loss 指淨損，net 指扣除相關項目後的淨額口徑。 | 866／13 |
| 2 | oper | 0.771615 | **營運損益欄**。財報用 Oper shr loss 和 Oper net loss 報營運損失。oper 標示營運口徑。 | 189／13 |
| 3 | 5.17 | 0.741170 | **低頻每股損失（意外）**。BROADVIEW 報兩期每股虧損 5.67 和 5.17 美元。後一數字在整個訓練集只見於兩篇。 | 2／5089 |
| 4 | profit | 0.739965 | **反向概念**。同一條每股財報可對照一季 loss 與另一季 profit。向量靠近反映共享句型，兩詞的盈虧方向相反。 | 569／13 |
| 5 | 31.09 | 0.732442 | **低頻每股損失（意外）**。FREEDOM SAVINGS 報每股虧損 31.09 美元。它在訓練集只出現兩次，這種近鄰依賴少量用例。 | 2／13192 |

**合理關係**　net、oper、vs 可與財報損益列的用法連結；profit 與 loss 有相反方向，兩者接近反映的可能是報表角色，而不是可互換的意思。

**新聞情境**　同一篇財報會並列本期損失、前期獲利、每股數值及營運結果。oper／net／vs 出現在相近句式中，容易形成共享的上下文。

**意外結果與可能原因**　571、1.40、5.17、31.09 等數字進入近鄰，說明小型語料中的金額或每股數字也可能共享損益模板。數字接近不是有效的金融詞義解釋，也不能據此判定文件檢索改善。

**原文證據示例**

| 證據詞 | 訓練 CF | 訓練 DF | 與分析詞共現篇數 |
|---|---:|---:|---:|
| oper | 932 | 260 | 189 |
| 1.40 | 18 | 18 | 5 |

證據詞：oper

訓練新聞 13：AM INTERNATIONAL INC <AM> 2ND QTR JAN 31

> AM INTERNATIONAL INC <AM> 2ND QTR JAN 31 Oper shr loss two cts vs profit seven cts     Oper shr profit 442,000 vs profit 2,986,000     Revs 291.8 mln vs 151.1 mln     Avg shrs 51.7 mln vs 43.4 mln     Six mths     Oper shr profit nil 

證據詞：1.40

訓練新聞 6202：REYNOLD METALS <RLM> UP ON FURMAN SELZ REPORT

> share for adoption of new accounting rules, 1.01 dlrs a share for tax loss carryforwards and other extraordinary items.  Reuter

訓練新聞 6202：REYNOLD METALS <RLM> UP ON FURMAN SELZ REPORT

>  holdings could be worth about 35 cts to 55 cts a share this year and 1.40 dlrs to two dlrs a share in 1988.     Reynolds Metals rose three points to 59-5/8 on volume of 729,500 shares.     Reynolds owns stakes in 

此處是文件共現，不是窗口內共現；共現為零時列出分開的原文情境，不能宣稱有直接共現。

### earnings

兩模型共有 0 個 Top-5 鄰近詞（無）。

#### CBOW

| 名次 | 鄰近詞 | cosine | 關係與解讀 | 共現篇數／訓練新聞 ID |
|---:|---|---:|---|---|
| 1 | profits | 0.659685 | **相近財務概念**。企業盈餘展望和實際季度獲利常在一篇財報中並列，概念接近且報告口徑可能不同。 | 78／23 |
| 2 | income | 0.631860 | **相近財務概念**。報導用 net income 估淨收益，再用 earnings 描述公司盈餘表現，兩者在此財報語境接近。 | 92／331 |
| 3 | revenues | 0.584460 | **同份財報的不同欄位**。IVACO 的新聞同時談首季盈餘和去年營收。營收在扣除費用前，與盈餘指標有別。 | 88／123 |
| 4 | results | 0.540163 | **較廣的財務結果**。operating results 可涵蓋銷售、獲利等營運結果。公司報告用它描述整體表現。 | 115／130 |
| 5 | losses | 0.480831 | **損益對照**。報導可同時交代持續營運的盈餘和待處分事業的虧損，兩者處在同一報表情境。 | 49／435 |

#### Skip-Gram

| 名次 | 鄰近詞 | cosine | 關係與解讀 | 共現篇數／訓練新聞 ID |
|---:|---|---:|---|---|
| 1 | 65.0 | 0.682235 | **低頻財報金額（意外）**。PREUSSAG 報導用 65.0 mln marks 描述淨利。這個數值在訓練集只出現五次。 | 3／3505 |
| 2 | minorco | 0.650297 | **公司專名（意外）**。MINORCO 自身發布盈餘展望。公司名可能靠近該事件用語，近鄰涵蓋範圍仍局限於少數報導。 | 2／8050 |
| 3 | telxon | 0.650070 | **公司專名（意外）**。TELXON 報導同時預估營收和每股盈餘。公司名只見於兩篇訓練新聞。 | 2／1315 |
| 4 | full-year | 0.649448 | **財報期間**。新聞會一起談全年、第四季與重編後盈餘。full-year 表示涵蓋期間。 | 8／2303 |
| 5 | thrid | 0.646374 | **疑似原文拼字錯誤（意外）**。原文寫 thrid quarter revenues and earnings，很可能是 third 的字母順序錯誤。詞只見於兩篇訓練新聞。 | 1／10096 |

**合理關係**　CBOW 的 profits、income、revenues、results 都與財務表現有關，但 revenues 與 earnings 不是完全相同的財務概念，losses 也不代表正向盈餘。

**新聞情境**　季度或全年財報通常一起談收入、盈餘及結果，full-year 也可能由這種報導期間語境而接近。這是財報情境上的關聯，並非所有近鄰都是同義詞。

**意外結果與可能原因**　Skip-Gram 的 65.0、minorco、telxon、thrid 與普通盈餘概念差異較大。thrid 在訓練新聞原文的季度敘述中出現，可能涉及原始拼寫及低頻效應；公司專名與數字也可能因少量相似句式而取得高分。下方 CF／DF 可核對低頻情形。

**原文證據示例**

| 證據詞 | 訓練 CF | 訓練 DF | 與分析詞共現篇數 |
|---|---:|---:|---:|
| income | 489 | 335 | 92 |
| thrid | 2 | 2 | 1 |

證據詞：income

訓練新聞 331：EXCO BUYS U.S. GOVERNMENT SECURITIES BROKER

>  Peter Goldie said RMJ would be bought at about the same multiple as Exco, suggesting net income of around 16 mln dlrs.     The company's earnings had not been hit by the halving of brokerage fees some 14 months ago as volumes had since doubled.     Lacy said that RMJ employed some 300 people, with 200 in the brokerage busin

證據詞：thrid

訓練新聞 10096：PDA ENGINEERING <PDAS> HALTS SOFTWARE SHIPMENTS

> o a technical problem with the program.     The company said the deferral will reduce its thrid quarter revenues and earnings.     PDA said it anticipates resuming shipments in the fourth quarter.  Reuter

此處是文件共現，不是窗口內共現；共現為零時列出分開的原文情境，不能宣稱有直接共現。

### acquisition

兩模型共有 4 個 Top-5 鄰近詞（transaction、purchase、merger、buyout）。

#### CBOW

| 名次 | 鄰近詞 | cosine | 關係與解讀 | 共現篇數／訓練新聞 ID |
|---:|---|---:|---|---|
| 1 | merger | 0.691048 | **同一領域的交易形式**。企業報告可在同篇提及買入股權與合併事件。acquisition 偏取得控制，merger 偏合併組織。 | 61／74 |
| 2 | acquisitions | 0.621937 | **同詞單複數**。Dean 的報導用複數說多筆收購，並用單數介紹其中一筆具名收購。 | 32／18 |
| 3 | transaction | 0.613391 | **較廣的交易稱呼**。Purolator 報導將股權收購與橋接融資合稱該筆交易。transaction 可涵蓋其他交易。 | 69／372 |
| 4 | purchase | 0.579135 | **取得資產的用詞**。Bally 標題用 purchase，內文用 acquisition 說同一次取得飯店。purchase 在其他新聞也可指一般購買。 | 91／447 |
| 5 | buyout | 0.567054 | **特定收購方式**。Purolator 的 buyout 新聞談現金買股。buyout 指買下股權或控制權，是較具體的收購情境。 | 26／372 |

#### Skip-Gram

| 名次 | 鄰近詞 | cosine | 關係與解讀 | 共現篇數／訓練新聞 ID |
|---:|---|---:|---|---|
| 1 | transaction | 0.644007 | **較廣的交易稱呼**。Purolator 報導將股權收購與橋接融資合稱該筆交易。transaction 可涵蓋其他交易。 | 69／372 |
| 2 | purchase | 0.638126 | **取得資產的用詞**。Bally 標題用 purchase，內文用 acquisition 說同一次取得飯店。purchase 在其他新聞也可指一般購買。 | 91／447 |
| 3 | merger | 0.624891 | **同一領域的交易形式**。企業報告可在同篇提及買入股權與合併事件。acquisition 偏取得控制，merger 偏合併組織。 | 61／74 |
| 4 | buyout | 0.617325 | **特定收購方式**。Purolator 的 buyout 新聞談現金買股。buyout 指買下股權或控制權，是較具體的收購情境。 | 26／372 |
| 5 | sahlen | 0.614532 | **低頻公司專名（意外）**。SAHLEN 報導談購入保全公司。它和 acquisition 沒有同篇紀錄，兩篇各自的購買新聞提供情境線索。 | 0／18、3673 |

**合理關係**　merger、acquisitions、transaction、purchase、buyout 都可連結企業交易或併購，兩模型的主要近鄰具有合理的事件情境關係。

**新聞情境**　併購新聞常在同一事件中描述交易、購買、合併與股權變動。purchase／transaction 比 acquisition 更廣，這種上下文接近不等於每個詞都能直接替換。

**意外結果與可能原因**　Skip-Gram 的 sahlen 是需回到新聞閱讀的專名，而非一般併購概念。它與 acquisition 在本批訓練文件的直接共現可能為零；此時只能提出相似上下文的可能性，不能宣稱有直接共現證據。

**原文證據示例**

| 證據詞 | 訓練 CF | 訓練 DF | 與分析詞共現篇數 |
|---|---:|---:|---:|
| merger | 601 | 287 | 61 |
| sahlen | 4 | 1 | 0 |

證據詞：merger

訓練新聞 74：ATICO FINANCIAL CORP <ATFC> 4TH QTR NET

>  vs 16.9 mln     Revs 45.00 mln vs 26.2 mln     NOTE: 1986 4th qtr and yr amounts include acquisition of 98.8 pct of common of Atico, formerly Peninsula Federal Savings and Loan Association, on January 24, 1986.     1985 4th qtr and yr net include net gain of 15.9 mln dlrs or 5.86 dlrs per share on exchange of common of Pan America Banks Inc for common of NCNB Corp purusant to a merger of the companies.  Reuter

證據詞：sahlen

訓練新聞 18：DEAN FOODS <DF> SEES STRONG 4TH QTR EARNINGS

> ons from Ryan Milk to the Larsen Co.     Dean also said the company will benefit from its acquisition in late December of Elgin Blenders Inc, West Chicago.     He said the company is a major shareholder of E.B.I. Foods Ltd, a United Kingdom blender, and has licensing arrangements 

訓練新聞 3673：SAHLEN AND ASSOCIATES <SALNU> COMPLETES PURCHASE

> SAHLEN AND ASSOCIATES <SALNU> COMPLETES PURCHASE Sahlen and Associates Inc said it has completed the purchase of Gleason Securities Service Inc of New York and Gleason Plant Security Inc

此處是文件共現，不是窗口內共現；共現為零時列出分開的原文情境，不能宣稱有直接共現。

### merger

兩模型共有 1 個 Top-5 鄰近詞（acquisition）。

#### CBOW

| 名次 | 鄰近詞 | cosine | 關係與解讀 | 共現篇數／訓練新聞 ID |
|---:|---|---:|---|---|
| 1 | acquisition | 0.691048 | **相鄰的併購事件**。同份公司報告能提到取得另一公司股權和兩家公司合併，兩個動作在併購新聞中常相連。 | 61／74 |
| 2 | transaction | 0.571210 | **較廣的交易稱呼**。Purolator 報導先交代合併對股東的安排，再用 transaction 指要完成的交易。 | 43／372 |
| 3 | buyout | 0.546119 | **股權買斷情境**。同篇 Purolator 報導以 buyout 作標題，內文描述交易中的合併與股東對價。 | 16／372 |
| 4 | kistler | 0.538669 | **合併案人物（意外）**。Kistler 是回應 Baker 合併案的 Hughes Tool 董事長。他的姓名跟隨少數事件進入近鄰。 | 3／2251 |
| 5 | reorganization | 0.534508 | **相鄰組織事件**。重組和合併都涉及公司結構。兩詞在本批訓練新聞沒有同篇紀錄，可從各自的企業調整報導理解。 | 0／74、10 |

#### Skip-Gram

| 名次 | 鄰近詞 | cosine | 關係與解讀 | 共現篇數／訓練新聞 ID |
|---:|---|---:|---|---|
| 1 | str | 0.658696 | **公司交易代號（意外）**。合併表決新聞用 <STR> 標記 Questar。代號靠近 merger 可由特定事件解讀。 | 2／7670 |
| 2 | rni | 0.647238 | **公司交易代號（意外）**。RANCO 標題中的 <RNI> 跟著股東通過合併的消息出現。這個代號只見於兩篇訓練新聞。 | 1／6499 |
| 3 | vons | 0.629724 | **合併案公司（意外）**。Allied 交易報導說條件取決於與 Vons 的待定合併。Vons 是事件參與者。 | 2／8355 |
| 4 | rainbow's | 0.625886 | **公司名稱的所有格（意外）**。原文用 Rainbow's planned merger 表示 Rainbow 規劃的合併。共享上下文可能來自少數具名交易。 | 3／7596 |
| 5 | acquisition | 0.624891 | **相鄰的併購事件**。同份公司報告能提到取得另一公司股權和兩家公司合併，兩個動作在併購新聞中常相連。 | 61／74 |

**合理關係**　CBOW 的 acquisition、transaction、buyout、reorganization 與合併或組織交易情境相關；reorganization 涵義較廣，不能一概視為 merger 的同義。

**新聞情境**　合併新聞會一起出現公司名稱、交易代號、股東批准與收購條件，因此模型也可能把特定參與公司的專名和 merger 放得接近。

**意外結果與可能原因**　Skip-Gram 的 str、rni、vons、rainbow's，以及 CBOW 的 kistler，較像特定事件中的專名或代號。RNI 的原文合併新聞可支持這個情境，但低頻專名的高分不代表通用的語意關係，也顯示兩架構的近鄰品質不能預設高下。

**原文證據示例**

| 證據詞 | 訓練 CF | 訓練 DF | 與分析詞共現篇數 |
|---|---:|---:|---:|
| acquisition | 697 | 502 | 61 |
| rni | 2 | 2 | 1 |

證據詞：acquisition

訓練新聞 74：ATICO FINANCIAL CORP <ATFC> 4TH QTR NET

>  vs 16.9 mln     Revs 45.00 mln vs 26.2 mln     NOTE: 1986 4th qtr and yr amounts include acquisition of 98.8 pct of common of Atico, formerly Peninsula Federal Savings and Loan Association, on January 24, 1986.     1985 4th qtr and yr net include net gain of 15.9 mln dlrs or 5.86 dlrs per share on exchange of common of Pan America Banks Inc for common of NCNB Corp purusant to a merger of the companies.  Reuter

證據詞：rni

訓練新聞 6499：RANCO <RNI> HOLDERS APPROVE MERGER

> RANCO <RNI> HOLDERS APPROVE MERGER Ranco Inc said shareholders at a special meeting approved a merger into <Siebe PLC> for 40 dlrs per share.  Reuter

此處是文件共現，不是窗口內共現；共現為零時列出分開的原文情境，不能宣稱有直接共現。

### oil

兩模型共有 2 個 Top-5 鄰近詞（crude、natural）。

#### CBOW

| 名次 | 鄰近詞 | cosine | 關係與解讀 | 共現篇數／訓練新聞 ID |
|---:|---|---:|---|---|
| 1 | kernel | 0.534936 | **棕櫚油的商品語境（意外）**。馬來西亞出口稅報導同時提到 palm oil 和 palm kernel。oil 可指食用油，這解釋它為何可能靠近棕櫚仁。 | 5／885 |
| 2 | natural | 0.518681 | **天然氣搭配**。阿根廷產量報導先列 crude oil，再列 natural gas。natural 的關係來自天然氣詞組。 | 83／708 |
| 3 | petroleum | 0.500589 | **石油語境近義**。oil 指石油時與 petroleum 意思接近。oil 還有食用油等其他用法。 | 192／140 |
| 4 | inter-city | 0.483376 | **油氣公司專名（意外）**。INTER-CITY GAS 財報提及油氣資產減記。inter-city 在訓練集僅出現兩次。 | 1／2483 |
| 5 | crude | 0.452701 | **原油搭配**。crude oil 指未精煉的原油。合約價格新聞以這個固定詞組報每桶價格。 | 216／127 |

#### Skip-Gram

| 名次 | 鄰近詞 | cosine | 關係與解讀 | 共現篇數／訓練新聞 ID |
|---:|---|---:|---|---|
| 1 | crude | 0.777472 | **原油搭配**。crude oil 指未精煉的原油。合約價格新聞以這個固定詞組報每桶價格。 | 216／127 |
| 2 | gas | 0.691155 | **同一產業的不同產品**。油氣設備或產量報導常寫 oil and gas。兩詞代表不同產品。 | 219／68 |
| 3 | shut-in | 0.634981 | **生產作業用語**。油稅新聞描述 shut-in production，也就是暫停生產。它描述油氣產量的變動方式。 | 3／8134 |
| 4 | liquids | 0.618474 | **油氣儲量分類**。ORBIT 儲量報導並列 oil 和 natural gas liquids。liquids 在此指天然氣液體詞組的一部分。 | 18／2007 |
| 5 | natural | 0.609948 | **天然氣搭配**。阿根廷產量報導先列 crude oil，再列 natural gas。natural 的關係來自天然氣詞組。 | 83／708 |

**合理關係**　Skip-Gram 的 crude、gas、shut-in、liquids、natural 與油氣生產情境相關；CBOW 的 petroleum 也有合理關係。gas／natural 等仍不是 oil 的完整同義詞。

**新聞情境**　oil 可出現在原油、天然氣，也可出現在棕櫚油等商品新聞。kernel 的共現新聞涉及棕櫚油與棕櫚仁出口稅，提供多義及商品情境的實例。

**意外結果與可能原因**　CBOW 將 kernel 排第一，inter-city 也進入 Top-5。前者可能反映食用油情境與少數共現，後者是需要核對的低頻專名；這些近鄰不應直接作為原油搜尋的同義擴展。

**原文證據示例**

| 證據詞 | 訓練 CF | 訓練 DF | 與分析詞共現篇數 |
|---|---:|---:|---:|
| kernel | 10 | 6 | 5 |
| inter-city | 2 | 2 | 1 |

證據詞：kernel

訓練新聞 885：MALAYSIA RAISES DUTY ON PROCESSED PALM OIL

> MALAYSIA RAISES DUTY ON PROCESSED PALM OIL The government said it raised the export duty on processed palm oil (PPO) to 64.06 ringgit per tonne from 40.96 ringgit, effective from March 1.     Export duty on crude palm oil (CPO) was unchanged at 16.06 ringgit per tonne.     The gazetted price of PPO rose to 796.8604 ringgit per tonne from 719.8286. That of CPO remained at 617.8238 ringgit.     The export duty and gazetted price of palm kernel were left unchanged at 191.15 and 955.75 ringgit per tonne respectively.  REUTER

證據詞：inter-city

訓練新聞 2483：INTER-CITY GAS CORP <ICG> 4TH QTR NET

> INTER-CITY GAS CORP <ICG> 4TH QTR NET Shr 36 cts vs 68 cts     Net 10.0 mln vs 16.1 mln     Revs 441.6 mln vs 470.8 mln     YEAR     Shr 86 cts vs 1.77 dlrs     Net 29.1 mln vs 44.1 mln     Revs 1.43 billion vs 1.54 billion     Note: 1986 fl-yr net includes 9.3 mln dlr writedown of U.S. oil and gas properties partly offset by 1.1 mln dlr extraordinary gain from tax gains and proceeds from sale of Minnesota utility operations. 1985 net includes extraordinary gain of 8

此處是文件共現，不是窗口內共現；共現為零時列出分開的原文情境，不能宣稱有直接共現。

### petroleum

兩模型共有 2 個 Top-5 鄰近詞（oil、gas）。

#### CBOW

| 名次 | 鄰近詞 | cosine | 關係與解讀 | 共現篇數／訓練新聞 ID |
|---:|---|---:|---|---|
| 1 | oil | 0.500589 | **石油語境近義**。石油公司報導可用 oil 指油品或油氣資產。oil 的日常用法仍比 petroleum 廣。 | 192／140 |
| 2 | energy | 0.490081 | **較廣的能源領域**。同一油市報導引用石油分析公司和能源政策研究者。energy 涵蓋石油以外的能源。 | 63／144 |
| 3 | esso | 0.484941 | **產業公司專名**。挪威勘探執照新聞並列 Esso 和多家 Petroleum 公司。這是企業與產業的關聯。 | 4／9279 |
| 4 | columbia | 0.464873 | **低共現的公司名稱片段（意外）**。共同出現的新聞標題是 British Columbia Resources，正文提及 Westar Petroleum 投資。兩詞只在一篇訓練新聞共現。 | 1／1456 |
| 5 | gas | 0.456656 | **同一產業的不同產品**。石油公司財報可一併討論油氣資產。gas 指氣體燃料或天然氣語境。 | 75／140 |

#### Skip-Gram

| 名次 | 鄰近詞 | cosine | 關係與解讀 | 共現篇數／訓練新聞 ID |
|---:|---|---:|---|---|
| 1 | oil | 0.596424 | **石油語境近義**。石油公司報導可用 oil 指油品或油氣資產。oil 的日常用法仍比 petroleum 廣。 | 192／140 |
| 2 | wpla | 0.579874 | **公司交易代號（意外）**。WOODSIDE PETROLEUM 標題把 <WPLA.S> 放在公司名旁。詞只有兩篇訓練用例。 | 2／11168 |
| 3 | gas | 0.571151 | **同一產業的不同產品**。石油公司財報可一併討論油氣資產。gas 指氣體燃料或天然氣語境。 | 75／140 |
| 4 | clnp | 0.563075 | **公司交易代號（意外）**。<CLNP> 跟在 CALLON PETROLEUM 公司名後。它識別公司，語境與該公司新聞綁定。 | 2／12151 |
| 5 | norte | 0.562987 | **油田地名片段（意外）**。管線報導提及 Cravo Norte 油田。norte 的近鄰可能反映具體產區或事件。 | 1／13184 |

**合理關係**　兩模型的 oil、gas，以及 CBOW 的 energy，都與能源情境相關，其中 oil 與 petroleum 較接近，但 gas／energy 涵義不同。

**新聞情境**　石油新聞中也包含企業、地名、交易代號與財務事件。esso、columbia、wpla、clnp、norte 等專名或代碼，可能因特定新聞語境而靠近 petroleum，需逐案閱讀。

**意外結果與可能原因**　wpla 的原文涉及 WOODSIDE PETROLEUM 與 WPLA.S，能支持特定公司代號的解讀。columbia 等專名也不能直接當成 petroleum 的同義詞；稀少共現或低頻關係不足以證明穩定語義。

**原文證據示例**

| 證據詞 | 訓練 CF | 訓練 DF | 與分析詞共現篇數 |
|---|---:|---:|---:|
| wpla | 2 | 2 | 2 |
| columbia | 41 | 24 | 1 |

證據詞：wpla

訓練新聞 11168：WOODSIDE PETROLEUM LTD <WPLA.S> 1986 YEAR

> WOODSIDE PETROLEUM LTD <WPLA.S> 1986 YEAR Shr nil vs same     Final and yr div nil vs same     Pre-tax, pre-minorities loss 3.53 mln dlrs vs profit 17.40 mln.     Net attributable loss 17.14 mln dlrs vs loss 

證據詞：columbia

訓練新聞 1456：B.C. RESOURCES HAS AGREEMENT ON CREDIT FACILITY

> ordinary charge also included a writedown of its investment in Westar Petroleum. Gains on the sales of Westar Timber's Skeena and Celgar pulp mills and Terrace sawmill partially offset the extraordinary loss, the compan

訓練新聞 1456：B.C. RESOURCES HAS AGREEMENT ON CREDIT FACILITY

> B.C. RESOURCES HAS AGREEMENT ON CREDIT FACILITY <British Columbia Resources Investment Corp>, earlier reporting higher full year operating losses, said it reached agreement in principle with five lenders p

此處是文件共現，不是窗口內共現；共現為零時列出分開的原文情境，不能宣稱有直接共現。

### wheat

兩模型共有 2 個 Top-5 鄰近詞（barley、grain）。

#### CBOW

| 名次 | 鄰近詞 | cosine | 關係與解讀 | 共現篇數／訓練新聞 ID |
|---:|---|---:|---|---|
| 1 | barley | 0.651183 | **同類作物**。農貸價格表把大麥與小麥列在同樣的價格欄位。兩者是不同穀物。 | 37／5 |
| 2 | corn | 0.630611 | **同類作物**。玉米與小麥在農貸、價格及產量表裡占相同欄位，形成相似的報導方式。 | 64／5 |
| 3 | soybeans | 0.582409 | **並列農產品**。一篇產量展望談小麥進口，再談黃豆及其他農作物產量。關聯來自農業情境，黃豆與小麥是不同作物。 | 27／235 |
| 4 | grains | 0.572603 | **上位類別的複數**。阿根廷出貨登記用 grains 統稱穀物，再列 bread wheat 等品種。 | 25／6 |
| 5 | grain | 0.570893 | **上位類別**。小麥是 grain 類別中的一種。農業制度報導也用 grain 統稱納入的品項。 | 82／5 |

#### Skip-Gram

| 名次 | 鄰近詞 | cosine | 關係與解讀 | 共現篇數／訓練新聞 ID |
|---:|---|---:|---|---|
| 1 | barley | 0.736833 | **同類作物**。農貸價格表把大麥與小麥列在同樣的價格欄位。兩者是不同穀物。 | 37／5 |
| 2 | durum | 0.710800 | **小麥品種**。價格表將 DURUM 與其他小麥品種排在 WHEAT 欄下，支持硬粒小麥的品種關係。 | 10／97 |
| 3 | winter | 0.679617 | **種植與季節**。小麥新聞會談生長季與越冬。例如原文在小麥病蟲害報導提到害蟲能熬過冬天。 | 39／241 |
| 4 | flour | 0.674438 | **加工產品**。出口補助新聞用 wheat flour 指小麥磨成的麵粉，形成原料與產品的搭配。 | 16／19 |
| 5 | grain | 0.669013 | **上位類別**。小麥是 grain 類別中的一種。農業制度報導也用 grain 統稱納入的品項。 | 82／5 |

**合理關係**　barley、corn、grain／grains 都與穀物相關；durum、winter、flour 可連結品種、季節或加工情境。這些是同領域或上下位／產業關係，並非全部等同小麥。

**新聞情境**　穀物價格表、出口與農業新聞常同時列出小麥、大麥、玉米及其他農產品，因此不同作物可以共享上下文。DURUM 的原文表格可作具體例子。

**意外結果與可能原因**　這組 Top-5 未見明顯跨到無關主題的詞，但 CBOW 的 soybeans 仍不是 wheat 的同義詞，winter 也需要搭配作物情境理解。這提醒我們合理的領域近鄰仍可能讓檢索範圍變寬；未出現明顯異常不等於模型已被證明正確。

**原文證據示例**

| 證據詞 | 訓練 CF | 訓練 DF | 與分析詞共現篇數 |
|---|---:|---:|---:|
| durum | 20 | 10 | 10 |
| soybeans | 129 | 71 | 27 |

證據詞：durum

訓練新聞 97：ASCS TERMINAL MARKET VALUES FOR PIK GRAIN

> ce per bushel is in U.S. dollars. Sorghum is priced per CWT, corn yellow grade only.      WHEAT     HRW   HRS    SRW   SWW   DURUM  Chicago       --    3.04   2.98  --    --  Ill. Track    --    --     3.16  --    --  Toledo        --    3.04   2.98  2.90  --  Memphis       --    --     3.05  --    --  Peoria

證據詞：soybeans

訓練新聞 235：INDONESIAN AGRICULTURE GROWTH EXPECTED TO SLOW

> , with a forecast 27.3 mln tonnes output in 1987.     The report says wheat imports are likely to fall to 1.5 mln tonnes in calendar 1987 from 1.69 mln tonnes in 1986 because of a drawdown on stocks.     "Growth pro

訓練新聞 235：INDONESIAN AGRICULTURE GROWTH EXPECTED TO SLOW

> ch is expected will be due to significant gains in production of corn soybeans, palm oil and palm kernels."     Constraints to significant overall increases in agricultural output include a shortage of disease resistan

此處是文件共現，不是窗口內共現；共現為零時列出分開的原文情境，不能宣稱有直接共現。

### grain

兩模型共有 3 個 Top-5 鄰近詞（wheat、grains、corn）。

#### CBOW

| 名次 | 鄰近詞 | cosine | 關係與解讀 | 共現篇數／訓練新聞 ID |
|---:|---|---:|---|---|
| 1 | grains | 0.620773 | **同詞單複數**。阿根廷登記報導以 grain 和 grains 表示穀物類別及其多種商品。 | 25／6 |
| 2 | corn | 0.606883 | **穀物品項**。玉米在農貸與穀物行情中作為具體品項，grain 表示較廣的類別。 | 59／5 |
| 3 | wheat | 0.570893 | **穀物品項**。小麥出現在穀物貸款價格表中。這是種類與類別的關係。 | 82／5 |
| 4 | soybeans | 0.532448 | **相鄰商品市場**。東京穀物交易所的報導談黃豆期貨保證金。相近來自交易場景，黃豆屬豆類油料作物。 | 23／229 |
| 5 | sugar | 0.523823 | **跨商品近鄰（意外）**。中國農業政策報導一同談糖料收購價和糧食配額。共享農業政策情境可能使不同商品靠近。 | 7／2975 |

#### Skip-Gram

| 名次 | 鄰近詞 | cosine | 關係與解讀 | 共現篇數／訓練新聞 ID |
|---:|---|---:|---|---|
| 1 | coarse | 0.678720 | **固定分類搭配**。暴雨報導寫 coarse grain harvesting。coarse 在詞組裡修飾穀物類別。 | 12／1582 |
| 2 | wheat | 0.669013 | **穀物品項**。小麥出現在穀物貸款價格表中。這是種類與類別的關係。 | 82／5 |
| 3 | grains | 0.666524 | **同詞單複數**。阿根廷登記報導以 grain 和 grains 表示穀物類別及其多種商品。 | 25／6 |
| 4 | corn | 0.652059 | **穀物品項**。玉米在農貸與穀物行情中作為具體品項，grain 表示較廣的類別。 | 59／5 |
| 5 | crop | 0.650804 | **較廣的農作物用語**。出貨登記報導說 crop registrations，涵蓋穀物和油籽等作物。 | 43／6 |

**合理關係**　grains、wheat、corn、crop 與穀物或農作物領域相關；coarse 在 coarse grain 的搭配中有直接情境依據。

**新聞情境**　穀物收成、出口及商品行情常並列多種作物，soybeans 等詞可能因農產品報導方式而接近。原文 coarse grain 的搭配支持的是用法關係，不是 coarse 單獨等於 grain。

**意外結果與可能原因**　CBOW 的 sugar 與穀物本身有類別差異，可能反映更廣泛的農產品／商品市場情境。Skip-Gram 的 coarse 也只是搭配詞；直接擴查可能引入不同商品，需由文件排名與 topics 評估另行檢驗。

**原文證據示例**

| 證據詞 | 訓練 CF | 訓練 DF | 與分析詞共現篇數 |
|---|---:|---:|---:|
| coarse | 21 | 12 | 12 |
| sugar | 624 | 149 | 7 |

證據詞：coarse

訓練新聞 1582：TORRENTIAL RAINS HALT ARGENTINE GRAIN HARVEST

> TORRENTIAL RAINS HALT ARGENTINE GRAIN HARVEST Torrential rains throughout Argentina's grain-producing areas virtually paralysed coarse grain harvesting in the week to yesterday, trade sources said.     Sunflower, maize and sorghum harvests were particularly affected, they said. But the rains proved to be a great 

證據詞：sugar

訓練新聞 2975：CHINA RAISES CROP PRICES TO INCREASE OUTPUT

> ES TO INCREASE OUTPUT China has raised the prices it pays farmers for cotton, edible oil, sugar cane and beets to reverse a decline in output in 1986, He Kang, Minister of Agriculture, Animal Husbandry and Fisheries said.     The China Daily quoted He as saying China should adopt intensive farming to increase per hectare output and improve crop quality and maintain arable land at 111 mln hectares. He gave no details of the price increases.     On grain, He said the state will cut the quota it purchases from farmers by 50 mln tonnes and abolish the practice of purchasing through agents.     He said the state will increase investm

此處是文件共現，不是窗口內共現；共現為零時列出分開的原文情境，不能宣稱有直接共現。

### trade

兩模型共有 0 個 Top-5 鄰近詞（無）。

#### CBOW

| 名次 | 鄰近詞 | cosine | 關係與解讀 | 共現篇數／訓練新聞 ID |
|---:|---|---:|---|---|
| 1 | warn | 0.467030 | **政策報導用語（意外）**。保護主義法案新聞用 warn 說明可能限制市場准入。這個動詞也能用在許多其他主題。 | 6／1201 |
| 2 | invisibles | 0.457420 | **相鄰的收支概念**。收支報導分開列 merchandise trade 與 invisibles 的逆差。後者是相鄰的非商品收支欄。 | 7／1890 |
| 3 | bilateral | 0.451902 | **雙邊貿易搭配**。美日談判新聞談 bilateral trade imbalance。bilateral 表示涉及雙方。 | 42／342 |
| 4 | retaliation | 0.442804 | **貿易政策反應**。紡織品進口限制新聞討論貿易夥伴可能報復。retaliation 表示政策行動的後果。 | 57／1197 |
| 5 | gatt | 0.441432 | **貿易協定名稱**。美日談判報導把 GATT 作為貿易協商框架，屬於同一制度情境。 | 58／342 |

#### Skip-Gram

| 名次 | 鄰近詞 | cosine | 關係與解讀 | 共現篇數／訓練新聞 ID |
|---:|---|---:|---|---|
| 1 | full-scale | 0.627403 | **會議規模修飾語（意外）**。原文用 full-scale 修飾 GATT conference。高分可能反映會議語境，這個形容詞可修飾許多活動。 | 5／8080 |
| 2 | surplus | 0.588324 | **貿易收支搭配**。巴西新聞用 trade surpluses 說明外貿收入。surplus 描述收支方向。 | 189／47 |
| 3 | aggravation | 0.586381 | **摩擦惡化的描述（意外）**。歐洲共同體聲明用 aggravation 描述貿易失衡加劇。三篇用例都與 trade 共現，證據範圍仍小。 | 3／5752 |
| 4 | tensions | 0.585891 | **貿易摩擦情境**。匯率變動報導用 trade tensions 指貿易緊張。tensions 也廣泛用於其他衝突。 | 16／1926 |
| 5 | deplore | 0.581505 | **外交聲明用語（意外）**。歐洲共同體用 deplore 批評貿易失衡加劇。它描述聲明語氣，只有少數訓練用例。 | 3／5752 |

**合理關係**　bilateral、retaliation、gatt、surplus 可與雙邊貿易、政策爭議或收支情境連結；它們描述不同面向，不能視為 trade 的同義詞。

**新聞情境**　貿易新聞常討論談判、報復、警告與國際緊張。invisibles 出現在收支新聞，full-scale 的原文是貿易會議敘述，說明普通修飾詞也可能透過相似情境接近。

**意外結果與可能原因**　warn、full-scale、aggravation、tensions、deplore 等詞較泛用，可能反映衝突報導的語氣，而非貿易主題的專屬詞義。這種上下文接近可能把匯率或國際收支新聞排前，Q09 的文件案例正好可檢驗這個風險，但不能據此證明單一詞造成了排名變動。

**原文證據示例**

| 證據詞 | 訓練 CF | 訓練 DF | 與分析詞共現篇數 |
|---|---:|---:|---:|
| invisibles | 11 | 7 | 7 |
| full-scale | 6 | 6 | 5 |

證據詞：invisibles

訓練新聞 1890：N.Z. CURRENT ACCOUNT DEFICIT NARROWS IN JANUARY

> on out random fluctuations -- shows a widening surplus on merchandise trade to 46 mln from a surplus of 33 mln, revised from 43 mln dlrs in December and a 71 mln deficit in the a year ago period.     Exports were 90

訓練新聞 1890：N.Z. CURRENT ACCOUNT DEFICIT NARROWS IN JANUARY

> ecember 1986 and a 323 mln deficit a year earlier.     The deficit on invisibles on unsmoothed figures eased to 215 mln dlrs from 248 mln, revised from 277 mln in December and 252 mln in January 1986.     The smoothed de

證據詞：full-scale

訓練新聞 8080：NEXT WORLD TRADE NEGOTIATIONS MUST SUCCEED - NZ

> NEXT WORLD TRADE NEGOTIATIONS MUST SUCCEED - NZ Ministers from more than 20 nations were told by New Zealand that the next international negotiations on lib

訓練新聞 8080：NEXT WORLD TRADE NEGOTIATIONS MUST SUCCEED - NZ

> " talks last Sepember. The meeting is also part of preparations for a full-scale conference of the General Agreement on Tariffs and Trade (GATT) in Venice in June.     The Uruguay meeting is considered by most countries 

此處是文件共現，不是窗口內共現；共現為零時列出分開的原文情境，不能宣稱有直接共現。

## 五 文件檢索效果

P@10 分母固定為 10；MRR 對每查詢完整返回排名的 RR 取平均；nDCG@10 使用二元 gain 與 log2 折扣。平均只排除沒有 topic 正例的查詢；有正例但查詢無向量則得零分。

| 方法 | P@10 | MRR | nDCG@10 | 查詢數 |
|---|---:|---:|---:|---:|
| TF-IDF | 0.8600 | 1.0000 | 0.8840 | 10/10 |
| TF-IDF 加權 CBOW | 0.9000 | 0.9000 | 0.8828 | 10/10 |
| TF-IDF 加權 Skip-Gram | 0.9300 | 0.9500 | 0.9266 | 10/10 |

### 逐查詢結果

| 查詢 | 方法 | P@10 | RR | nDCG@10 |
|---|---|---:|---:|---:|
| Q01 | TF-IDF | 0.8000 | 1.0000 | 0.8630 |
| Q01 | TF-IDF 加權 CBOW | 1.0000 | 1.0000 | 1.0000 |
| Q01 | TF-IDF 加權 Skip-Gram | 1.0000 | 1.0000 | 1.0000 |
| Q02 | TF-IDF | 0.9000 | 1.0000 | 0.9216 |
| Q02 | TF-IDF 加權 CBOW | 1.0000 | 1.0000 | 1.0000 |
| Q02 | TF-IDF 加權 Skip-Gram | 1.0000 | 1.0000 | 1.0000 |
| Q03 | TF-IDF | 1.0000 | 1.0000 | 1.0000 |
| Q03 | TF-IDF 加權 CBOW | 1.0000 | 1.0000 | 1.0000 |
| Q03 | TF-IDF 加權 Skip-Gram | 1.0000 | 1.0000 | 1.0000 |
| Q04 | TF-IDF | 1.0000 | 1.0000 | 1.0000 |
| Q04 | TF-IDF 加權 CBOW | 0.9000 | 1.0000 | 0.8611 |
| Q04 | TF-IDF 加權 Skip-Gram | 1.0000 | 1.0000 | 1.0000 |
| Q05 | TF-IDF | 1.0000 | 1.0000 | 1.0000 |
| Q05 | TF-IDF 加權 CBOW | 1.0000 | 1.0000 | 1.0000 |
| Q05 | TF-IDF 加權 Skip-Gram | 1.0000 | 1.0000 | 1.0000 |
| Q06 | TF-IDF | 0.4000 | 1.0000 | 0.4784 |
| Q06 | TF-IDF 加權 CBOW | 0.7000 | 0.5000 | 0.6004 |
| Q06 | TF-IDF 加權 Skip-Gram | 1.0000 | 1.0000 | 1.0000 |
| Q07 | TF-IDF | 1.0000 | 1.0000 | 1.0000 |
| Q07 | TF-IDF 加權 CBOW | 1.0000 | 1.0000 | 1.0000 |
| Q07 | TF-IDF 加權 Skip-Gram | 1.0000 | 1.0000 | 1.0000 |
| Q08 | TF-IDF | 1.0000 | 1.0000 | 1.0000 |
| Q08 | TF-IDF 加權 CBOW | 1.0000 | 1.0000 | 1.0000 |
| Q08 | TF-IDF 加權 Skip-Gram | 1.0000 | 1.0000 | 1.0000 |
| Q09 | TF-IDF | 0.8000 | 1.0000 | 0.8201 |
| Q09 | TF-IDF 加權 CBOW | 0.8000 | 0.5000 | 0.7015 |
| Q09 | TF-IDF 加權 Skip-Gram | 0.8000 | 0.5000 | 0.6699 |
| Q10 | TF-IDF | 0.7000 | 1.0000 | 0.7574 |
| Q10 | TF-IDF 加權 CBOW | 0.6000 | 1.0000 | 0.6652 |
| Q10 | TF-IDF 加權 Skip-Gram | 0.5000 | 1.0000 | 0.5965 |

## 六 改善與失敗案例

### Q06 改善案例與局部失敗：石油產出價格

查詢：petroleum output prices；目標 topic：crude。

| 方法 | P@10 | RR | nDCG@10 |
|---|---:|---:|---:|
| TF-IDF | 0.4000 | 1.0000 | 0.4784 |
| TF-IDF 加權 CBOW | 0.7000 | 0.5000 | 0.6004 |
| TF-IDF 加權 Skip-Gram | 1.0000 | 1.0000 | 1.0000 |

**已觀察到的事實**

- P@10 由 TF-IDF 的 0.40，變為加權 CBOW 的 0.70、加權 Skip-Gram 的 1.00；CBOW 的 RR 卻由 1.00 降至 0.50。改善前十名相關篇數，並不保證改善第一篇相關文件的位置。
- 17046 是 BRITISH PETROLEUM 的股權要約簡訊，topics 為 acq，TF-IDF 排第 3。20008 是原油增產新聞，topics 為 crude，從 TF-IDF 第 4 移到 CBOW 第 2、Skip-Gram 第 1。
- CBOW 的第 1 篇 17441 是汽柴油零售價格新聞，topics 為 fuel／gas；第 3 篇 18280 是銅價新聞，topics 為 copper。這些都是可閱讀並核對的局部失敗，不是整體指標必然改善的證明。

| 新聞 ID | 標題 | topics | TF-IDF 名次 | CBOW 名次 | Skip-Gram 名次 | 共同查詢詞 |
|---|---|---|---:|---:|---:|---|
| 17046 | STANDARD OIL SAYS BRITISH PETROLEUM SHARE TENDER EXTENDED UNTIL MAY 4 | acq | 3 | 431 | 144 | petroleum |
| 20031 | BANGLADESH TENDERS FOR TWO MLN BARRELS PETROLEUM | jet | 2 | 71 | 59 | petroleum |
| 20008 | MALAYSIA ADVISED TO RAISE CRUDE OIL OUTPUT IN 1988 | crude | 4 | 2 | 1 | output, prices |
| 18523 | INDONESIA CUTS JULY CRUDE SUPPLIES TO JAPAN | crude | 309 | 5 | 2 | prices |
| 17441 | SINGAPORE OIL COMPANIES TO SET OWN PUMP PRICES | fuel, gas | 33 | 1 | 20 | prices |
| 18280 | PHELPS DODGE <PD> SEES STRONGER COPPER PRICES | copper | 85 | 3 | 69 | petroleum, prices |

**可能原因**　詞彙匹配可能把 petroleum 作為公司名稱的短訊排前；加權詞向量則可能讓原油生產與價格語境的新聞靠近查詢。另一方面，一般商品價格、產量與能源語境也可能讓 CBOW 偏向燃油或銅價新聞。這些是結合原文與排名的解釋推論；本案例沒有消融實驗，不能宣稱已證明單一詞或同義替換是改善原因。

**可推論範圍**　本文依目標 crude topic 計分。fuel／gas 新聞在一般語意下仍可能與石油價格有關，但不屬於本次 crude 正例集合；不把這個主題代理改稱人工語意正確率。

**新聞 17046 原文（僅標題）**

> STANDARD OIL SAYS BRITISH PETROLEUM SHARE TENDER EXTENDED UNTIL MAY 4

**新聞 20031 原文（標題與正文）**

> BANGLADESH TENDERS FOR TWO MLN BARRELS PETROLEUM Bangladesh Petroleum Corp said it floated an international tender for imports of two mln barrels of Jet Kero, Superior Kero and High Speed Diesel for shipment during January-June 1988.     It said the offer for the petroleum products would be open until 0600 gmt on November 19.  REUTER

**新聞 20008 原文（標題與正文）**

> MALAYSIA ADVISED TO RAISE CRUDE OIL OUTPUT IN 1988 Malaysia's national oil company, Petronas, has advised the government to raise crude oil output to 540,000 barrels a day (bpd) in 1988 from a current 500,000 bpd, a senior company official said.     "We have the capacity to produce the amount," Rastam Hadi, Petronas's Vice-President for Upstream Sector said.     The government will announce its decision on Friday when it unveils the country's budget. Malaysia raised output this month to current levels from 420,000 bpd after reviewing the world oil market. In May, Malaysia cut output to 420,000 bpd from 459,000 in response to a call by OPEC to boost prices.  REUTER

**新聞 18523 原文（標題與正文）**

> INDONESIA CUTS JULY CRUDE SUPPLIES TO JAPAN Indonesia's state oil company, Pertamina, has informed its affiliated Japanese companies that crude oil supplies will be 40 pct less than contractual volumes in July, a spokesman for an affiliated company said.     He said the allocations were in line with Indonesia's production ceiling under its current OPEC quota, so they might be increased if OPEC ratifies a production rise next week.     Allocations of Indonesia's main traded crude, Minas, had been cut by 50 pct and Duri by 38 pct, he said.     Last month contractual volumes were cut by 30 pct.     Contractual volumes assume no production ceiling.     Oil traders said Indonesian grades were already trading above their official selling prices (osp) on the spot market due to strong demand from Japan and the U.S. And tight supplies.     A cargo of Duri loading in July was reportedly traded yesterday at 70 cents over its osp of 15.60 dlrs per barrel and Minas is trading about 20 cents higher, they said.  REUTER

**新聞 17441 原文（標題與正文）**

> SINGAPORE OIL COMPANIES TO SET OWN PUMP PRICES Singapore oil companies can set their own petrol and diesel prices at the pump from today subject to ceilings determined by their individual wholesale prices, The Ministry of Trade and Industry said.     The ministry previously revised pump prices and announced the changes for the oil companies.     Lead content of petrol is further cut to 0.15 gm/litre from 0.4 gm from today. Pump prices of the lower lead petrol grades are expected to be announced soon by the oil companies which have said wholesale prices are between 1.3 cents/litre to 3.2 cents higher than previous grades.  REUTER

**新聞 18280 原文（標題與正文節錄）**

> PHELPS DODGE <PD> SEES STRONGER COPPER PRICES Phelps Dodge Corp officials said good fundamentals in copper markets should lead to improving prices for the metal.     In an interview with Reuters, chairman G. Robert Durham said continued strong demand and low inventories pushed prices up eight to nine pct on the New York Commodities Exchange last month.     "Our customers in this country are living off the tailgates of our trucks," he said, referring to tightness of supply and strong demand. "The fundamentals are good."         Asked if metal prices will continue to rise, Durham said; "All I know is, fundamentals cannot be ignored."     He said copper supplies are lower than they have been for almost 20 years. Last year, copper demand was second highest on record behind 1984, he said.     Phelps Dodge is the nation's largest copper company, with annual production expected to reach 500,000 tons this year.         During the interview, executive vice president Douglas Yearley said he believed it was only a matter of time before the copper price rose "because there aren't that many new projects coming on, and demand, short of a major recession, will continue to grow modestly."     Copper for July delivery rose more than two cts a pound to more than 69 cents in Comex trading today. A one cent a pound rise in copper prices yields 10 mln dlrs in annual earnings for Phelps Dodge, the c

### Q09 失敗案例：前十名篇數相同但排序變差

查詢：international trade deficit；目標 topic：trade。

| 方法 | P@10 | RR | nDCG@10 |
|---|---:|---:|---:|
| TF-IDF | 0.8000 | 1.0000 | 0.8201 |
| TF-IDF 加權 CBOW | 0.8000 | 0.5000 | 0.7015 |
| TF-IDF 加權 Skip-Gram | 0.8000 | 0.5000 | 0.6699 |

**已觀察到的事實**

- 三方法的 P@10 都是 0.80，但 TF-IDF 的 RR 為 1.00，兩種 embedding 都是 0.50；nDCG@10 也從約 0.8201 降到 CBOW 的 0.7015、Skip-Gram 的 0.6699。
- 15352 的內容討論預算赤字政策是否能改善貿易逆差，topics 含 trade；它由 TF-IDF 第 1 移到 CBOW 第 13、Skip-Gram 第 22。
- 兩種 embedding 的第 1 篇都是 20082，其標題涉及美元下跌與貿易逆差，但原始 topics 只有 money-fx。17916 談經常帳逆差、topics 為 bop，也在 Skip-Gram 第 3。

| 新聞 ID | 標題 | topics | TF-IDF 名次 | CBOW 名次 | Skip-Gram 名次 | 共同查詢詞 |
|---|---|---|---:|---:|---:|---|
| 15352 | DEFICIT CUTS SEEN UNABLE TO CURE TRADE DEFICIT | trade | 1 | 13 | 22 | deficit, trade |
| 20082 | Further marked dollar fall would burden U.S. Trade deficit -Stoltenberg | money-fx | 4 | 1 | 1 | deficit, trade |
| 17916 | FINLAND'S CURRENT ACCOUNT IN DEFICIT IN 1ST QTR | bop | 5 | 6 | 3 | deficit |
| 19062 | JAPAN MUST TRY HARDER TO CUT SURPLUS - MINISTER | trade | 303 | 3 | 2 | trade |

**可能原因**　trade deficit、美元匯率與國際收支共享經濟新聞上下文，加權平均可能使這些相鄰主題接近，削弱了詞彙匹配對特定用語的區分。已量到的是依 topics 規則的排序退步，並非證明 20082 在自然語意上完全不相關；它只有標題文字，不能補造不存在的正文支持判斷。

**可推論範圍**　這個案例同時暴露粗粒度、多標籤 topics 的邊界。應保留既定參考標籤及原始指標，不因看到標題合理就改寫正例；語意解讀與資料集計分分開呈現。

**新聞 15352 原文（標題與正文節錄）**

> DEFICIT CUTS SEEN UNABLE TO CURE TRADE DEFICIT Financial analysts say they are pleased with congressional moves to trim next year's federal budget deficit but believe the actions will do little to help improve the U.S. trade deficit or buoy the economy.     The House of Representatives is expected to vote tomorrow to approve a trillion-dollar budget blueprint for the coming fiscal year that reduces the deficit by 38 billion dlrs.     Similarly, the Senate Budget Committee has approved a plan that would cut federal red ink by about 37 billion dlrs next year.         "In terms of the economy, 37-38 billion dlrs is infinitesimal, so cuts of this magnitude will have little impact on the economy and the trade deficit," said Stanley Collander, a Touche Ross federal budget policy analyst.     "At best, it will have a small positive effect," Collander said in an interview.     Federal Reserve Board Chairman Paul Volcker has repeatedly told Congress that cutting federal red ink would go a long way to help reduce the massive trade deficit and also help ease some of the downward pressure on the value of the dollar.     The U.S. government has attempted to remedy the trade imbalance by driving down the value of the dollar. But Volcker has warned that a further fall in the dollar's value is fraught with danger.     Such a decline, he has said, could refuel inflation as imported goods become

**新聞 20082 原文（僅標題）**

> Further marked dollar fall would burden U.S. Trade deficit -Stoltenberg

**新聞 17916 原文（標題與正文）**

> FINLAND'S CURRENT ACCOUNT IN DEFICIT IN 1ST QTR Finland's current account balance of payments showed a deficit of 2.5 billion markka in first quarter 1987 against a deficit of 3.2 billion in the same period last year, the Bank of Finland said in a statement.  REUTER

**新聞 19062 原文（標題與正文）**

> JAPAN MUST TRY HARDER TO CUT SURPLUS - MINISTER Japan must try harder to reduce its trade surplus or the yen will come under renewed upward pressure, Economic Planning Minister Tetsuo Kondo.     He told a press luncheon he hoped Japan could cut its current account surplus by some 10 billion dlrs a year until it is reduced to some 50 to 60 billion dlrs. In the fiscal year ended March 31 the surplus totalled 93.76 billion dlrs, or about 4.5 pct of GNP.     The 6,000 billion yen economic stimulation package the government unveiled late last month would help slash Japan's current account surplus by five to six billion dlrs, he said.  REUTER

### Q01 解讀邊界：獲利查詢也會召回虧損新聞

查詢：corporate profit earnings；目標 topic：earn。

| 方法 | P@10 | RR | nDCG@10 |
|---|---:|---:|---:|
| TF-IDF | 0.8000 | 1.0000 | 0.8630 |
| TF-IDF 加權 CBOW | 1.0000 | 1.0000 | 1.0000 |
| TF-IDF 加權 Skip-Gram | 1.0000 | 1.0000 | 1.0000 |

**已觀察到的事實**

- 三方法都使用 earn 作為目標 topic；此查詢的完整指標列在下表，不把任一份財報自行改成人工負例。
- 20419 的季度淨利報表是 TF-IDF 第 1；15979 的年度與季度財務展望，在兩種 embedding 都是第 1。
- 16711 談預期季度虧損及前期獲利比較，topics 為 earn，從 TF-IDF 第 333 移到 CBOW 第 2。它是主題正例，但不能僅憑 earn 標籤宣稱滿足『正向獲利』這種更細的查詢意圖。

| 新聞 ID | 標題 | topics | TF-IDF 名次 | CBOW 名次 | Skip-Gram 名次 | 共同查詢詞 |
|---|---|---|---:|---:|---:|---|
| 20419 | TEXAS EASTERN CORP <TET> 3RD QTR NET | earn | 1 | 38 | 62 | earnings, profit |
| 15979 | WESTINGHOUSE ELECTRIC <WX> OPTIMISTIC FOR YEAR | earn | 185 | 1 | 1 | earnings, profit |
| 16711 | CONVERGENT TECHNOLOGIES <CVGT.O> SEES QTR LOSS | earn | 333 | 2 | 29 | profit |

**可能原因**　profit／loss 在相似財報句式中接近，可能使加權文件向量保留財務主題，卻不充分區分損益方向、預期或不同期間。新聞原文可支持『內容方向不同』的觀察；不能以一組詞的近鄰分數直接證明 16711 排名上升的成因。

**可推論範圍**　原查詢是 corporate profit earnings，沒有另外定義『只找正向獲利』的人工作業標準。本案例用來展示主題與細節意圖的差別，不變更查詢、topics 或既有評分。

**新聞 20419 原文（標題與正文）**

> TEXAS EASTERN CORP <TET> 3RD QTR NET Shr profit three cts vs profit 36 cts     Net profit 1,800,000 vs profit 19.2 mln     Nine mths     Shr profit 1.22 dlrs vs loss 44 cts     Net 64.7 mln vs loss 23.5 mln     NOTE: 1986 3rd qtr and nine months includes a profit of 2.8 mln dlrs and a loss of 66.7 mln dlrs from discontinued operations.     Earnings per share are reported after payment of preferred stock dividends of subsidiaries.  Reuter

**新聞 15979 原文（標題與正文）**

> WESTINGHOUSE ELECTRIC <WX> OPTIMISTIC FOR YEAR Westinghouse Electric Corp chairman Douglas Danforth said he was encouraged by first quarter results for fiscal 1987, and said the company was positioned to capitalize on the economy's modest growth for the rest of the year.     "I am encouraged by the continuing improvement in earnings," Danworth said. "The first quarter met our expectations."     Westinghouse recorded net income for the first quarter of 151.6 mln dlrs, or 1.05 dlrs per share, on revenues of 2.32 billion dlrs.     Danworth attributed the results to an increase in the operating profit in the energy and advanced technology, industries and commerical segments.  Reuter

**新聞 16711 原文（標題與正文）**

> CONVERGENT TECHNOLOGIES <CVGT.O> SEES QTR LOSS Convergent Technologies Inc said it expects to report in the first quarter a loss more than twice the size of the 4.8-mln-dlr loss reported in the fourth quarter of 1986.     Convergent reported a first quarter 1986 profit of 2,100,000 dlrs, or five cts per share.     The company said results declined in the quarter both in its traditional OEM business and its business systems group.     The anticipated loss reflects lower than expected operating margins, start-up costs for new product manufacturing and higher than planned expenses.  Reuter

## 七 限制與結論

本次 Skip-Gram 的 P@10 與 nDCG@10 較高，但 TF-IDF 的 MRR 較高；CBOW 在 Q06 的 P@10 改善，同時仍有燃油／銅價新聞排前的局部失敗。Q09 表明相同 P@10 不等於相同排序品質。
profit／loss 的近鄰與 Q01 說明：共享財務情境不等於相同損益方向。加權平均不完整保留詞序、否定、時態與複雜語意；保留否定詞本身也不代表理解了否定。
topics 是粗粒度且可能多標籤的參考，不能把主題未命中直接等同自然語意錯誤。類別不平衡、每主題兩查詢的依賴、低頻詞、單 seed 與小型語料限制一般化，沒有顯著性或大型語言模型能力的推論。
詞近鄰、文件共現、PCA、loss 與檢索指標分別描述不同事物。文件案例的原因解讀沒有消融實驗支持，不宣稱單一近鄰或某個詞已被證明造成改善。

## 八 重現與來源

程式、獨立虛擬環境與資料位於 IR-HW4，未依賴 HW3 執行環境。原始模型與 evaluation-v1 實驗保留；本報告只從它們衍生，完整綁定與檔案 SHA256 見同目錄 manifest.json。

執行 scripts/test_hw4.ps1 可測試四頁及計算；scripts/verify_hw4.py 重算數值。scripts/build_report.py 另建新的報告版本，scripts/verify_analysis.py 核對詞頻、原文片段、案例名次及整合報告。網站只讀完成的報告，不在 rerun 重新建檔。
語料 SHA256：e086d597efcc71d9399414c7264cece1d285ebdda65619d33b882cccced3408f
模型 manifest SHA256：fc7753bf476b9384afe4fe0a46fc02281faaca9830aab1b6a01b6bcbdbbb6845
原評估 manifest SHA256：ed65f01de38f0d808fe75b877b735b193459cd992bac0a355fed1b3cce3945cd

[Reuters-21578 官方原始 README](https://kdd.ics.uci.edu/databases/reuters21578/README.txt)
[Gensim Word2Vec 訓練 API](https://radimrehurek.com/gensim/models/word2vec.html)

## 附錄 三方法對十查詢的完整 Top 10

| 查詢 | 方法 | 名次 | 新聞 ID | cosine | 目標 topic 命中 | 標題 |
|---|---|---:|---|---:|---:|---|
| Q01 | tfidf | 1 | 20419 | 0.240080 | 1 | TEXAS EASTERN CORP <TET> 3RD QTR NET |
| Q01 | tfidf | 2 | 21208 | 0.232744 | 1 | HOMETOWN BANCORP INC <HTWN.O> 3RD QTR NET |
| Q01 | tfidf | 3 | 15797 | 0.210401 | 1 | SILICON SYSTEMS INC <SLCN> 2ND QTR MARCH 28 |
| Q01 | tfidf | 4 | 15188 | 0.206953 | 1 | JUDY'S INC <JUDY> 4TH QTR JAN 31 LOSS |
| Q01 | tfidf | 5 | 21106 | 0.200414 | 1 | WASHINGTON WATER POWER <WWP> 3RD QTR LOSS |
| Q01 | tfidf | 6 | 21175 | 0.199410 | 1 | CITYFED FINANCIAL CORP <CTYF.O> 3RD QTR LOSS |
| Q01 | tfidf | 7 | 19828 | 0.197959 | 0 | SAN FRANCISCO, NOT REGION, HURT BY RESTRUCTURING |
| Q01 | tfidf | 8 | 20927 | 0.189167 | 1 | <ACC CORP> 3RD QTR NET |
| Q01 | tfidf | 9 | 21047 | 0.187664 | 1 | MERCURY SAVINGS AND LOAN <MSL> 3RD QTR LOSS |
| Q01 | tfidf | 10 | 17651 | 0.187450 | 0 | INSURANCE FIRM AS PIEZO <PEPI.O> PREFERRED STAKE |
| Q01 | cbow | 1 | 15979 | 0.668556 | 1 | WESTINGHOUSE ELECTRIC <WX> OPTIMISTIC FOR YEAR |
| Q01 | cbow | 2 | 16711 | 0.646607 | 1 | CONVERGENT TECHNOLOGIES <CVGT.O> SEES QTR LOSS |
| Q01 | cbow | 3 | 14967 | 0.636865 | 1 | RUBBERMAID <RBD> SEES CONTINUED IMPROVEMENT |
| Q01 | cbow | 4 | 16824 | 0.625339 | 1 | BUSH INDUSTRIES <BSH> SEES HIGHER YEAR RESULTS |
| Q01 | cbow | 5 | 16336 | 0.607879 | 1 | NCR <NCR> CONTINUES TO EXPECT EARNINGS GROWTH |
| Q01 | cbow | 6 | 15641 | 0.604490 | 1 | ABBOTT <ABT> SEES GAINS FROM WEAKER DOLLAR |
| Q01 | cbow | 7 | 16893 | 0.604101 | 1 | FHLBB REPORTS THRIFT RESULTS IN FOURTH QUARTER |
| Q01 | cbow | 8 | 15263 | 0.601101 | 1 | GENERAL ELECTRIC <GE> 1ST QTR HELPED BY RCA |
| Q01 | cbow | 9 | 16343 | 0.597834 | 1 | MERRILL LYNCH <MER> INVESTMENT REVENUES UP |
| Q01 | cbow | 10 | 18514 | 0.591437 | 1 | LONDON INTERNATIONAL RAISE PROFITS BY 3.05 MLN |
| Q01 | skipgram | 1 | 15979 | 0.732356 | 1 | WESTINGHOUSE ELECTRIC <WX> OPTIMISTIC FOR YEAR |
| Q01 | skipgram | 2 | 15263 | 0.718054 | 1 | GENERAL ELECTRIC <GE> 1ST QTR HELPED BY RCA |
| Q01 | skipgram | 3 | 16336 | 0.709142 | 1 | NCR <NCR> CONTINUES TO EXPECT EARNINGS GROWTH |
| Q01 | skipgram | 4 | 14967 | 0.696453 | 1 | RUBBERMAID <RBD> SEES CONTINUED IMPROVEMENT |
| Q01 | skipgram | 5 | 15598 | 0.692669 | 1 | T. ROWE PRICE <TROW> SEES HIGHER FIRST QUARTER |
| Q01 | skipgram | 6 | 16893 | 0.688771 | 1 | FHLBB REPORTS THRIFT RESULTS IN FOURTH QUARTER |
| Q01 | skipgram | 7 | 16824 | 0.685108 | 1 | BUSH INDUSTRIES <BSH> SEES HIGHER YEAR RESULTS |
| Q01 | skipgram | 8 | 16312 | 0.684361 | 1 | IRVING BANK <V> CITES LOANS IN EARNING DECLINE |
| Q01 | skipgram | 9 | 16644 | 0.684148 | 1 | BANK OF NEW ENGLAND <BKNE.O> POST PRO FORMA NET |
| Q01 | skipgram | 10 | 15103 | 0.681458 | 1 | J.P. MORGAN <JPM> NET HURT BY BRAZIL, TRADING |
| Q02 | tfidf | 1 | 15759 | 0.213290 | 1 | FITCHBURG GAS <FGE> RAISES QUARTERLY |
| Q02 | tfidf | 2 | 15457 | 0.212001 | 1 | UNIFIRST CORP <UNF> SETS QUARTERLY |
| Q02 | tfidf | 3 | 15702 | 0.211840 | 1 | MEDITRUST <MTRUS> RAISES QUARTERLY |
| Q02 | tfidf | 4 | 16687 | 0.207510 | 1 | MONTGOMERY STREET INCOME <MTS> 1ST QTR NET |
| Q02 | tfidf | 5 | 16834 | 0.193805 | 1 | GORMAN-RUPP CO <GRC> SETS QUARTERLY |
| Q02 | tfidf | 6 | 18664 | 0.192872 | 0 | U.S. PERSONAL INCOME ROSE 0.2 PCT IN MAY, SPENDING UP 0.1 PCT |
| Q02 | tfidf | 7 | 20304 | 0.191537 | 1 | BANKERS TRUST <BT> 3RD QTR NET |
| Q02 | tfidf | 8 | 16803 | 0.191502 | 1 | FIRST FEDERAL BANK <FFBN.O> 2ND QTR MARCH 31 NET |
| Q02 | tfidf | 9 | 16246 | 0.189576 | 1 | INTERNATIONAL BUSINESS MACHINES CORP <IBM> NET |
| Q02 | tfidf | 10 | 16434 | 0.188510 | 1 | IRVING BANK <V> 1ST-QTR NET HURT BY BRAZIL |
| Q02 | cbow | 1 | 16472 | 0.678063 | 1 | FRANKLIN PUERTO RICO SETS MONTHLY PAYOUT |
| Q02 | cbow | 2 | 16470 | 0.674425 | 1 | FRANKLIN INSURED SETS MONTHLY PAYOUT |
| Q02 | cbow | 3 | 16469 | 0.667119 | 1 | FRANKLIN MINNESOTA SETS MONTHLY PAYOUT |
| Q02 | cbow | 4 | 16717 | 0.664566 | 1 | <TRAVELERS REALTY INCOME INVESTORS> 1ST QTR NET |
| Q02 | cbow | 5 | 16473 | 0.664444 | 1 | FRANKLIN CALIFORNIA SETS MONTHLY PAYOUT |
| Q02 | cbow | 6 | 20685 | 0.660612 | 1 | NL INDUSTRIES INC <NL> 3RD QTR NET |
| Q02 | cbow | 7 | 15190 | 0.658562 | 1 | DOW JONES AND CO INC <DJ> 1ST QTR NET |
| Q02 | cbow | 8 | 16468 | 0.656444 | 1 | FRANKLIN MASSACHUSETTS SETS MONTHLY PAYOUT |
| Q02 | cbow | 9 | 16471 | 0.642749 | 1 | FRANKLIN OHIO SETS MONTHLY PAYOUT |
| Q02 | cbow | 10 | 15759 | 0.638452 | 1 | FITCHBURG GAS <FGE> RAISES QUARTERLY |
| Q02 | skipgram | 1 | 16470 | 0.757093 | 1 | FRANKLIN INSURED SETS MONTHLY PAYOUT |
| Q02 | skipgram | 2 | 16473 | 0.755331 | 1 | FRANKLIN CALIFORNIA SETS MONTHLY PAYOUT |
| Q02 | skipgram | 3 | 16469 | 0.754626 | 1 | FRANKLIN MINNESOTA SETS MONTHLY PAYOUT |
| Q02 | skipgram | 4 | 16394 | 0.747563 | 1 | FRANKLIN PENNSYLVANIA TAX-FREE IN INITIAL PAYOUT |
| Q02 | skipgram | 5 | 16468 | 0.740881 | 1 | FRANKLIN MASSACHUSETTS SETS MONTHLY PAYOUT |
| Q02 | skipgram | 6 | 16393 | 0.740246 | 1 | FRANKLIN HIGH YIELD SETS HIGHER PAYOUT |
| Q02 | skipgram | 7 | 15280 | 0.738439 | 1 | HANOVER INSURANCE <HINS> RAISES DIVIDEND |
| Q02 | skipgram | 8 | 16392 | 0.737862 | 1 | FRANKLIN MICHIGAN INSURED SETS LOWER PAYOUT |
| Q02 | skipgram | 9 | 15655 | 0.736183 | 1 | HEALTH/REHABILITATION <HRP> SETS FIRST PAYOUT |
| Q02 | skipgram | 10 | 15593 | 0.734628 | 1 | TEXAS AMERICAN <TAE> OMITS PREFERRED PAYOUT |
| Q03 | tfidf | 1 | 16583 | 0.374254 | 1 | <FIRST MONTAUK SECURITIES> IN MERGER DEAL |
| Q03 | tfidf | 2 | 19796 | 0.293296 | 1 | SETON <SEL> DIRECTORS APPROVE MERGER AGREEMENT |
| Q03 | tfidf | 3 | 20835 | 0.292190 | 1 | UNITED ARTISTS <UACI.O> SUSPENDS MERGER TALKS |
| Q03 | tfidf | 4 | 19240 | 0.282789 | 1 | STIFEL FINANCIAL SAID IT RECEIVED UNSOLICITED MERGER PROPOSAL |
| Q03 | tfidf | 5 | 18239 | 0.277492 | 1 | WESBANCO <WSBC.O> ACQUIRING BANK OF SISSONVILLE |
| Q03 | tfidf | 6 | 17683 | 0.261317 | 1 | HICKS AND HAAS GROUP GETS FINANCING FOR SPECTRADYNE ACQUISITION |
| Q03 | tfidf | 7 | 18682 | 0.259484 | 1 | USAIR <U> SAYS DOT PUBLIC COUNSEL BACKS MERGER |
| Q03 | tfidf | 8 | 16033 | 0.258844 | 1 | CADILLAC FAIRVIEW SAYS IT HAS RECEIVED SOME ACQUISITION PROPOSALS |
| Q03 | tfidf | 9 | 16465 | 0.237049 | 1 | ARRAYS <ARAY> COMPLETES MERGER |
| Q03 | tfidf | 10 | 16041 | 0.201973 | 1 | WINTERHALTER <WNTLC> HOLDERS OKAY TAKEOVER |
| Q03 | cbow | 1 | 16465 | 0.804439 | 1 | ARRAYS <ARAY> COMPLETES MERGER |
| Q03 | cbow | 2 | 19796 | 0.752743 | 1 | SETON <SEL> DIRECTORS APPROVE MERGER AGREEMENT |
| Q03 | cbow | 3 | 18861 | 0.747957 | 1 | MAGELLAN SETS MERGER WITH BALZAC INVESTMENTS |
| Q03 | cbow | 4 | 19281 | 0.716703 | 1 | NORTH AMERICAN COMMUNICATIONS WEIGHS POSSIBLE SALE OR MERGER |
| Q03 | cbow | 5 | 20835 | 0.698576 | 1 | UNITED ARTISTS <UACI.O> SUSPENDS MERGER TALKS |
| Q03 | cbow | 6 | 16676 | 0.697235 | 1 | INTERCO <ISS> COMPLETES LANE<LANE> ACQUISITION |
| Q03 | cbow | 7 | 16583 | 0.694393 | 1 | <FIRST MONTAUK SECURITIES> IN MERGER DEAL |
| Q03 | cbow | 8 | 17787 | 0.688600 | 1 | KINGSBRIDGE, MASCO SET MERGER |
| Q03 | cbow | 9 | 17589 | 0.685536 | 1 | (SUBURBAN BANCORP), (WOODSTOCK BANCORP) TO MERGE |
| Q03 | cbow | 10 | 19325 | 0.680183 | 1 | N. AMERICAN COMMUNICATIONS <NACS.O> WEIGHS SALE |
| Q03 | skipgram | 1 | 16465 | 0.814087 | 1 | ARRAYS <ARAY> COMPLETES MERGER |
| Q03 | skipgram | 2 | 19796 | 0.741909 | 1 | SETON <SEL> DIRECTORS APPROVE MERGER AGREEMENT |
| Q03 | skipgram | 3 | 18861 | 0.725159 | 1 | MAGELLAN SETS MERGER WITH BALZAC INVESTMENTS |
| Q03 | skipgram | 4 | 21478 | 0.718230 | 1 | GAF CORP SAID MANAGEMENT GROUP TO RECONSIDER ACQUISITION PROPOSAL |
| Q03 | skipgram | 5 | 15060 | 0.712272 | 1 | SCI-MED <SMLS> BOARD AGREES TO BRISTOL<BMY>DEAL |
| Q03 | skipgram | 6 | 16041 | 0.710044 | 1 | WINTERHALTER <WNTLC> HOLDERS OKAY TAKEOVER |
| Q03 | skipgram | 7 | 14943 | 0.703859 | 1 | CONRAC CORP SAID IT HAS ENTERED TALKS ON ITS ACQUISITION WITH SEVERAL PARTIES |
| Q03 | skipgram | 8 | 16583 | 0.703378 | 1 | <FIRST MONTAUK SECURITIES> IN MERGER DEAL |
| Q03 | skipgram | 9 | 16714 | 0.693938 | 1 | INTERCO <ISS> SHAREHOLDERS APPROVE MERGER |
| Q03 | skipgram | 10 | 19240 | 0.691352 | 1 | STIFEL FINANCIAL SAID IT RECEIVED UNSOLICITED MERGER PROPOSAL |
| Q04 | tfidf | 1 | 18959 | 0.402260 | 1 | NOVA CORP SAYS IT IS CONSIDERING A TAKEOVER BID FOR DOME PETROLEUM |
| Q04 | tfidf | 2 | 19963 | 0.371589 | 1 | STE-GENEVIEVE UPS OFFER FOR SULLIVAN <SUM.TO> |
| Q04 | tfidf | 3 | 19900 | 0.272679 | 1 | TECSYN INTERNATIONAL TERMINATES TAKEOVER TALKS WITH INVESTOR GROUP |
| Q04 | tfidf | 4 | 16493 | 0.266058 | 1 | NOR-QUEST <NQRLF> MAKES BID FOR NORTHAIR <NRM.TO> |
| Q04 | tfidf | 5 | 17876 | 0.248552 | 1 | EQUITICORP TASMAN TO BID FOR MONIER LTD |
| Q04 | tfidf | 6 | 19912 | 0.242347 | 1 | TECSYN <TSNA.TO> TERMINATES TAKEOVER TALKS |
| Q04 | tfidf | 7 | 19378 | 0.241846 | 1 | REED INTL DENIES BID APPROACH |
| Q04 | tfidf | 8 | 16688 | 0.229617 | 1 | NORTHAIR <NRM.TO> OPPOSING NORQUEST <NQRLF> BID |
| Q04 | tfidf | 9 | 21003 | 0.225497 | 1 | CCR VIDEO SAYST RECEIVED OFFER TO NEGOTIATE A TAKEOVER BY INTERCEP INVESTMENT CORP |
| Q04 | tfidf | 10 | 19748 | 0.202200 | 1 | SEC CHARGES CAMPEAU'S ALLIED STORES WITH DISCLOSURE VIOLATIONS IN RECENT TAKEOVER |
| Q04 | cbow | 1 | 18585 | 0.740226 | 1 | SERVICE RESOURCES <SRC> ENDS SORG <SRG> BID |
| Q04 | cbow | 2 | 20773 | 0.726224 | 0 | DART GROUP DROPS DAYTON HUDSON <DH> TAKEOVER BID |
| Q04 | cbow | 3 | 19963 | 0.715678 | 1 | STE-GENEVIEVE UPS OFFER FOR SULLIVAN <SUM.TO> |
| Q04 | cbow | 4 | 18566 | 0.708431 | 1 | CSR DECLARES PIONEER SUGAR BID UNCONDITIONAL |
| Q04 | cbow | 5 | 17876 | 0.685572 | 1 | EQUITICORP TASMAN TO BID FOR MONIER LTD |
| Q04 | cbow | 6 | 20489 | 0.680544 | 1 | GOODMAN FIELDER HAS 29.9 PCT OF RANKS HOVIS |
| Q04 | cbow | 7 | 15600 | 0.677851 | 1 | GENERAL PARTNERS CUTS STAKE IN GENCORP <GY> |
| Q04 | cbow | 8 | 16145 | 0.666860 | 1 | WILLIAMS DETAILS ACCEPTANCES OF NORCROS OFFER |
| Q04 | cbow | 9 | 20379 | 0.660435 | 1 | GORDON TO MAKE BID TO ACQUIRE PAGECORP <PGOA.TO> |
| Q04 | cbow | 10 | 19631 | 0.654815 | 1 | MARTIN PROCESSING SAID IT ACCEPTS 20 DLRS/SHARE BID FROM COURTAULDS PLC |
| Q04 | skipgram | 1 | 19900 | 0.713665 | 1 | TECSYN INTERNATIONAL TERMINATES TAKEOVER TALKS WITH INVESTOR GROUP |
| Q04 | skipgram | 2 | 20321 | 0.699839 | 1 | ICAHN DROPS BID FOR TWA <TWA> |
| Q04 | skipgram | 3 | 19470 | 0.695185 | 1 | WPP GROUP PLC EXTENDS ITS OFFER FOR JWT <JWT> |
| Q04 | skipgram | 4 | 20306 | 0.693298 | 1 | TRANS WORLD AIRLINES SAID CARL ICAHN WITHDRAWS ACQUISITION OFFER |
| Q04 | skipgram | 5 | 18566 | 0.692657 | 1 | CSR DECLARES PIONEER SUGAR BID UNCONDITIONAL |
| Q04 | skipgram | 6 | 16037 | 0.689690 | 1 | CADILLAC FAIRVIEW SAYS IT RECEIVED TAKEOVER BIDS |
| Q04 | skipgram | 7 | 19912 | 0.688930 | 1 | TECSYN <TSNA.TO> TERMINATES TAKEOVER TALKS |
| Q04 | skipgram | 8 | 19364 | 0.688264 | 1 | IRWIN JACOBS HAS GILLETTE <GS> STAKE |
| Q04 | skipgram | 9 | 19631 | 0.685593 | 1 | MARTIN PROCESSING SAID IT ACCEPTS 20 DLRS/SHARE BID FROM COURTAULDS PLC |
| Q04 | skipgram | 10 | 17876 | 0.685018 | 1 | EQUITICORP TASMAN TO BID FOR MONIER LTD |
| Q05 | tfidf | 1 | 18746 | 0.349919 | 1 | UNION PACIFIC <UNP> RAISES CRUDE OIL PRICES |
| Q05 | tfidf | 2 | 17875 | 0.322777 | 1 | SOUTH KOREA INCREASES DUTY ON CRUDE IMPORTS |
| Q05 | tfidf | 3 | 18698 | 0.317078 | 1 | IMPERIAL OIL <IMO.A> RAISES CRUDE 32 CANADIAN CTS |
| Q05 | tfidf | 4 | 18736 | 0.309819 | 1 | MURPHY OIL <MUR> RAISES CRUDE POSTINGS |
| Q05 | tfidf | 5 | 18773 | 0.309002 | 1 | PETRO-CANADA CRUDE UP 32 CTS CANADIAN/BBL |
| Q05 | tfidf | 6 | 18678 | 0.305228 | 1 | IMPERIAL OIL RAISES CRUDE OIL POSTINGS 32 CANADIAN CTS/BBL, LIGHT SWEET NOW 25.60 |
| Q05 | tfidf | 7 | 20420 | 0.295119 | 1 | PHILLIPS <P> RAISES CRUDE OIL PRICES |
| Q05 | tfidf | 8 | 18810 | 0.292964 | 1 | DIAMOND SHAMROCK <DRM> RAISES CRUDE POSTINGS |
| Q05 | tfidf | 9 | 16658 | 0.281200 | 1 | ABU DHABI MARKETING SAID NOT BREACHING OPEC PACT |
| Q05 | tfidf | 10 | 18655 | 0.272424 | 1 | SHELL CANADA RAISES CRUDE OIL POSTING 32 CANADIAN CTS/BBL |
| Q05 | cbow | 1 | 20008 | 0.767587 | 1 | MALAYSIA ADVISED TO RAISE CRUDE OIL OUTPUT IN 1988 |
| Q05 | cbow | 2 | 18108 | 0.761648 | 1 | ENERGY SECRETARY SEES HIGHER OIL PRICES |
| Q05 | cbow | 3 | 18146 | 0.741430 | 1 | SMALL RISE SEEN IN U.S. GASOLINE INVENTORIES |
| Q05 | cbow | 4 | 18523 | 0.736260 | 1 | INDONESIA CUTS JULY CRUDE SUPPLIES TO JAPAN |
| Q05 | cbow | 5 | 18493 | 0.735376 | 1 | AUSTRALIAN CRUDE OUTPUT FALLS IN FIRST NINE MONTHS |
| Q05 | cbow | 6 | 19110 | 0.734373 | 1 | SAUDI OIL RESERVES RISE DESPITE HIGHER OUTPUT |
| Q05 | cbow | 7 | 16005 | 0.724575 | 1 | CRUDE OIL NETBACKS UP SHARPLY IN EUROPE, U.S. |
| Q05 | cbow | 8 | 18066 | 0.722372 | 1 | U.S. ENERGY SECRETARY SEES HIGHER OIL PRICES |
| Q05 | cbow | 9 | 16658 | 0.718668 | 1 | ABU DHABI MARKETING SAID NOT BREACHING OPEC PACT |
| Q05 | cbow | 10 | 20270 | 0.716503 | 1 | NORWAY BOOSTS OIL OUTPUT BY 22 PCT IN SEPTEMBER |
| Q05 | skipgram | 1 | 16005 | 0.760522 | 1 | CRUDE OIL NETBACKS UP SHARPLY IN EUROPE, U.S. |
| Q05 | skipgram | 2 | 18773 | 0.759328 | 1 | PETRO-CANADA CRUDE UP 32 CTS CANADIAN/BBL |
| Q05 | skipgram | 3 | 20008 | 0.755727 | 1 | MALAYSIA ADVISED TO RAISE CRUDE OIL OUTPUT IN 1988 |
| Q05 | skipgram | 4 | 18678 | 0.731475 | 1 | IMPERIAL OIL RAISES CRUDE OIL POSTINGS 32 CANADIAN CTS/BBL, LIGHT SWEET NOW 25.60 |
| Q05 | skipgram | 5 | 20944 | 0.729418 | 1 | ARCO RAISES CRUDE OIL PRICES 50 CTS BARREL, TODAY, WTI TO 19.00 |
| Q05 | skipgram | 6 | 18707 | 0.728167 | 1 | PERMIAN RAISES CRUDE OIL POSTINGS 50 CTS  A BBL, WTI TO 19 DLRS |
| Q05 | skipgram | 7 | 20420 | 0.727468 | 1 | PHILLIPS <P> RAISES CRUDE OIL PRICES |
| Q05 | skipgram | 8 | 19110 | 0.726931 | 1 | SAUDI OIL RESERVES RISE DESPITE HIGHER OUTPUT |
| Q05 | skipgram | 9 | 17875 | 0.726388 | 1 | SOUTH KOREA INCREASES DUTY ON CRUDE IMPORTS |
| Q05 | skipgram | 10 | 19193 | 0.725335 | 1 | TEXACO <TXC> CANADA TO RAISE CRUDE OIL POSTINGS |
| Q06 | tfidf | 1 | 17478 | 0.274085 | 1 | PEMEX OFFICIAL SAYS OPEC OUTPUT CRITICAL TO PRICE |
| Q06 | tfidf | 2 | 20031 | 0.232539 | 0 | BANGLADESH TENDERS FOR TWO MLN BARRELS PETROLEUM |
| Q06 | tfidf | 3 | 17046 | 0.232347 | 0 | STANDARD OIL SAYS BRITISH PETROLEUM SHARE TENDER EXTENDED UNTIL MAY 4 |
| Q06 | tfidf | 4 | 20008 | 0.221498 | 1 | MALAYSIA ADVISED TO RAISE CRUDE OIL OUTPUT IN 1988 |
| Q06 | tfidf | 5 | 18493 | 0.212829 | 1 | AUSTRALIAN CRUDE OUTPUT FALLS IN FIRST NINE MONTHS |
| Q06 | tfidf | 6 | 19509 | 0.212415 | 1 | NORWAY EXTENDS 7.5 PCT OIL OUTPUT CUT - MINISTRY |
| Q06 | tfidf | 7 | 14854 | 0.195865 | 0 | CHINA INDUSTRIAL OUTPUT RISES IN FIRST QUARTER |
| Q06 | tfidf | 8 | 19082 | 0.193590 | 0 | GERMAN PRODUCER PRICES RISE 0.1 PCT IN MAY |
| Q06 | tfidf | 9 | 16897 | 0.187664 | 0 | SOVIET INDUSTRIAL OUTPUT UP IN FIRST QUARTER |
| Q06 | tfidf | 10 | 18959 | 0.183666 | 0 | NOVA CORP SAYS IT IS CONSIDERING A TAKEOVER BID FOR DOME PETROLEUM |
| Q06 | cbow | 1 | 17441 | 0.655689 | 0 | SINGAPORE OIL COMPANIES TO SET OWN PUMP PRICES |
| Q06 | cbow | 2 | 20008 | 0.653018 | 1 | MALAYSIA ADVISED TO RAISE CRUDE OIL OUTPUT IN 1988 |
| Q06 | cbow | 3 | 18280 | 0.649135 | 0 | PHELPS DODGE <PD> SEES STRONGER COPPER PRICES |
| Q06 | cbow | 4 | 17478 | 0.647960 | 1 | PEMEX OFFICIAL SAYS OPEC OUTPUT CRITICAL TO PRICE |
| Q06 | cbow | 5 | 18523 | 0.643823 | 1 | INDONESIA CUTS JULY CRUDE SUPPLIES TO JAPAN |
| Q06 | cbow | 6 | 18146 | 0.633786 | 1 | SMALL RISE SEEN IN U.S. GASOLINE INVENTORIES |
| Q06 | cbow | 7 | 18108 | 0.632093 | 1 | ENERGY SECRETARY SEES HIGHER OIL PRICES |
| Q06 | cbow | 8 | 19083 | 0.631619 | 0 | JAPAN FIRMS TO LAUNCH SALES OF 100 OCTANE GASOLINE |
| Q06 | cbow | 9 | 19509 | 0.630379 | 1 | NORWAY EXTENDS 7.5 PCT OIL OUTPUT CUT - MINISTRY |
| Q06 | cbow | 10 | 19998 | 0.628350 | 1 | VENEZUELA SEES FLAT OIL PRICE DESPITE U.S. ATTACK |
| Q06 | skipgram | 1 | 20008 | 0.765141 | 1 | MALAYSIA ADVISED TO RAISE CRUDE OIL OUTPUT IN 1988 |
| Q06 | skipgram | 2 | 18523 | 0.747784 | 1 | INDONESIA CUTS JULY CRUDE SUPPLIES TO JAPAN |
| Q06 | skipgram | 3 | 18493 | 0.739468 | 1 | AUSTRALIAN CRUDE OUTPUT FALLS IN FIRST NINE MONTHS |
| Q06 | skipgram | 4 | 19509 | 0.730960 | 1 | NORWAY EXTENDS 7.5 PCT OIL OUTPUT CUT - MINISTRY |
| Q06 | skipgram | 5 | 19998 | 0.723546 | 1 | VENEZUELA SEES FLAT OIL PRICE DESPITE U.S. ATTACK |
| Q06 | skipgram | 6 | 19499 | 0.721995 | 1 | FAR EAST TRADERS EXPECT 20 DLR OIL IN JANUARY |
| Q06 | skipgram | 7 | 18146 | 0.718663 | 1 | SMALL RISE SEEN IN U.S. GASOLINE INVENTORIES |
| Q06 | skipgram | 8 | 19559 | 0.717309 | 1 | KUWAIT SEES OPEC OUTPUT OVER CEILING TO END-YEAR |
| Q06 | skipgram | 9 | 19110 | 0.715950 | 1 | SAUDI OIL RESERVES RISE DESPITE HIGHER OUTPUT |
| Q06 | skipgram | 10 | 16005 | 0.715002 | 1 | CRUDE OIL NETBACKS UP SHARPLY IN EUROPE, U.S. |
| Q07 | tfidf | 1 | 20645 | 0.378791 | 1 | USDA REPORTS CORN SOLD TO USSR |
| Q07 | tfidf | 2 | 15893 | 0.332037 | 1 | SOVIET UNION TO IMPORT MORE GRAIN IN 86/87-USDA |
| Q07 | tfidf | 3 | 15999 | 0.328488 | 1 | GRAIN SHIPMENTS TO THE USSR -- USDA |
| Q07 | tfidf | 4 | 15856 | 0.311792 | 1 | USSR ADDS U.S. CORN TO COMMITMENTS - USDA |
| Q07 | tfidf | 5 | 18297 | 0.305223 | 1 | U.S. WHEAT IN SENEGAL MARKET -- USDA |
| Q07 | tfidf | 6 | 18954 | 0.300904 | 1 | CHINA ADDS CORN TO COMMITMENTS - USDA |
| Q07 | tfidf | 7 | 16765 | 0.271153 | 1 | 60,000 TONNES CORN SMUGGLED INTO PHILIPPINES-PAPER |
| Q07 | tfidf | 8 | 16525 | 0.263023 | 1 | U.S. CORN EXPORTS SEEN WELL AHEAD OF LAST YEAR |
| Q07 | tfidf | 9 | 15845 | 0.257619 | 1 | USDA - U.S. 1986/87 ENDING CORN STOCKS  5,240 MLN BU, WHEAT 1,848 MLN, SOYBEANS 610 MLN |
| Q07 | tfidf | 10 | 16012 | 0.249753 | 1 | EGYPT SEEKING 500,000 TONNES CORN - U.S. TRADERS |
| Q07 | cbow | 1 | 18297 | 0.795196 | 1 | U.S. WHEAT IN SENEGAL MARKET -- USDA |
| Q07 | cbow | 2 | 15893 | 0.794703 | 1 | SOVIET UNION TO IMPORT MORE GRAIN IN 86/87-USDA |
| Q07 | cbow | 3 | 21091 | 0.778200 | 1 | U.S. EXPORT INSPECTIONS, IN THOUS BUSHELS,  SOYBEANS 16,333 WHEAT 30,917 CORN 36,781 |
| Q07 | cbow | 4 | 18954 | 0.759012 | 1 | CHINA ADDS CORN TO COMMITMENTS - USDA |
| Q07 | cbow | 5 | 17722 | 0.753846 | 1 | U.S. EXPORT INSPECTIONS, IN THOUS BUSHELS,  SOYBEANS 7,209, WHEAT 15,187 CORN 25,347 |
| Q07 | cbow | 6 | 19821 | 0.745572 | 1 | U.S. EXPORT INSPECTIONS, IN THOUS BUSHELS, SOYBEANS 8,392 WHEAT 33,641 CORN 33,728 |
| Q07 | cbow | 7 | 20637 | 0.743936 | 1 | U.S. EXPORTERS REPORT 125,000 TONNES OF  CORN SOLD TO USSR FOR 1987/88 |
| Q07 | cbow | 8 | 15856 | 0.742858 | 1 | USSR ADDS U.S. CORN TO COMMITMENTS - USDA |
| Q07 | cbow | 9 | 15999 | 0.741761 | 1 | GRAIN SHIPMENTS TO THE USSR -- USDA |
| Q07 | cbow | 10 | 20645 | 0.732788 | 1 | USDA REPORTS CORN SOLD TO USSR |
| Q07 | skipgram | 1 | 15893 | 0.801623 | 1 | SOVIET UNION TO IMPORT MORE GRAIN IN 86/87-USDA |
| Q07 | skipgram | 2 | 20637 | 0.771276 | 1 | U.S. EXPORTERS REPORT 125,000 TONNES OF  CORN SOLD TO USSR FOR 1987/88 |
| Q07 | skipgram | 3 | 15999 | 0.763561 | 1 | GRAIN SHIPMENTS TO THE USSR -- USDA |
| Q07 | skipgram | 4 | 15911 | 0.763057 | 1 | USDA REPORTS EXPORT SALES ACTIVITY |
| Q07 | skipgram | 5 | 20645 | 0.753766 | 1 | USDA REPORTS CORN SOLD TO USSR |
| Q07 | skipgram | 6 | 15845 | 0.750603 | 1 | USDA - U.S. 1986/87 ENDING CORN STOCKS  5,240 MLN BU, WHEAT 1,848 MLN, SOYBEANS 610 MLN |
| Q07 | skipgram | 7 | 15856 | 0.749228 | 1 | USSR ADDS U.S. CORN TO COMMITMENTS - USDA |
| Q07 | skipgram | 8 | 15904 | 0.746489 | 1 | U.S. EXPORTERS REPORT 100,000 TONNES CORN  SOLD TO ALGERIA FOR 1986/87 DELIVERY |
| Q07 | skipgram | 9 | 18954 | 0.730233 | 1 | CHINA ADDS CORN TO COMMITMENTS - USDA |
| Q07 | skipgram | 10 | 18297 | 0.727141 | 1 | U.S. WHEAT IN SENEGAL MARKET -- USDA |
| Q08 | tfidf | 1 | 18297 | 0.335466 | 1 | U.S. WHEAT IN SENEGAL MARKET -- USDA |
| Q08 | tfidf | 2 | 15893 | 0.329978 | 1 | SOVIET UNION TO IMPORT MORE GRAIN IN 86/87-USDA |
| Q08 | tfidf | 3 | 14841 | 0.265921 | 1 | SRI LANKA GETS USDA APPROVAL FOR WHEAT PRICE |
| Q08 | tfidf | 4 | 15582 | 0.260071 | 1 | YUGOSLAVIA TO TENDER FOR 100,000 TONNES WHEAT |
| Q08 | tfidf | 5 | 15916 | 0.253035 | 1 | EXPORT BONUS WHEAT FOR JORDAN --USDA |
| Q08 | tfidf | 6 | 15388 | 0.252079 | 1 | CHINA'S WHEAT CROP THREATENED BY PESTS, DISEASE |
| Q08 | tfidf | 7 | 19322 | 0.245757 | 1 | BRAZIL BUYS FRENCH WHEAT AT TENDER |
| Q08 | tfidf | 8 | 15866 | 0.236099 | 1 | USDA ESTIMATES ARGENTINE COARSE GRAIN, WHEAT |
| Q08 | tfidf | 9 | 21530 | 0.232606 | 1 | CHINA FORECASTS 397 MLN TONNES GRAIN HARVEST |
| Q08 | tfidf | 10 | 15726 | 0.226906 | 1 | LONDON GRAIN FREIGHTS |
| Q08 | cbow | 1 | 15893 | 0.806563 | 1 | SOVIET UNION TO IMPORT MORE GRAIN IN 86/87-USDA |
| Q08 | cbow | 2 | 18297 | 0.799326 | 1 | U.S. WHEAT IN SENEGAL MARKET -- USDA |
| Q08 | cbow | 3 | 15582 | 0.772993 | 1 | YUGOSLAVIA TO TENDER FOR 100,000 TONNES WHEAT |
| Q08 | cbow | 4 | 21123 | 0.729407 | 1 | SOVIET 1988 GRAIN TARGET INDICATES 1987 OUTPUT |
| Q08 | cbow | 5 | 21091 | 0.718831 | 1 | U.S. EXPORT INSPECTIONS, IN THOUS BUSHELS,  SOYBEANS 16,333 WHEAT 30,917 CORN 36,781 |
| Q08 | cbow | 6 | 15999 | 0.718334 | 1 | GRAIN SHIPMENTS TO THE USSR -- USDA |
| Q08 | cbow | 7 | 15388 | 0.715083 | 1 | CHINA'S WHEAT CROP THREATENED BY PESTS, DISEASE |
| Q08 | cbow | 8 | 15911 | 0.712776 | 1 | USDA REPORTS EXPORT SALES ACTIVITY |
| Q08 | cbow | 9 | 21530 | 0.710170 | 1 | CHINA FORECASTS 397 MLN TONNES GRAIN HARVEST |
| Q08 | cbow | 10 | 16144 | 0.698043 | 1 | AWB CHAIRMAN URGES FARMERS NOT TO CUT PLANTINGS |
| Q08 | skipgram | 1 | 15893 | 0.828439 | 1 | SOVIET UNION TO IMPORT MORE GRAIN IN 86/87-USDA |
| Q08 | skipgram | 2 | 15911 | 0.797362 | 1 | USDA REPORTS EXPORT SALES ACTIVITY |
| Q08 | skipgram | 3 | 15999 | 0.784559 | 1 | GRAIN SHIPMENTS TO THE USSR -- USDA |
| Q08 | skipgram | 4 | 18297 | 0.779280 | 1 | U.S. WHEAT IN SENEGAL MARKET -- USDA |
| Q08 | skipgram | 5 | 15582 | 0.774532 | 1 | YUGOSLAVIA TO TENDER FOR 100,000 TONNES WHEAT |
| Q08 | skipgram | 6 | 15863 | 0.762254 | 1 | USDA ESTIMATES SOVIET WHEAT, COARSE GRAINS |
| Q08 | skipgram | 7 | 21123 | 0.760968 | 1 | SOVIET 1988 GRAIN TARGET INDICATES 1987 OUTPUT |
| Q08 | skipgram | 8 | 15866 | 0.759457 | 1 | USDA ESTIMATES ARGENTINE COARSE GRAIN, WHEAT |
| Q08 | skipgram | 9 | 15856 | 0.752261 | 1 | USSR ADDS U.S. CORN TO COMMITMENTS - USDA |
| Q08 | skipgram | 10 | 20637 | 0.751024 | 1 | U.S. EXPORTERS REPORT 125,000 TONNES OF  CORN SOLD TO USSR FOR 1987/88 |
| Q09 | tfidf | 1 | 15352 | 0.410979 | 1 | DEFICIT CUTS SEEN UNABLE TO CURE TRADE DEFICIT |
| Q09 | tfidf | 2 | 16125 | 0.344850 | 1 | SINGAPORE EXTERNAL TRADE GAINS 8.8 PCT IN QUARTER |
| Q09 | tfidf | 3 | 17452 | 0.331661 | 1 | U.S. STUDY SAYS TARIFFS AND QUOTAS COULD BACKFIRE |
| Q09 | tfidf | 4 | 20082 | 0.309040 | 0 | Further marked dollar fall would burden U.S. Trade deficit -Stoltenberg |
| Q09 | tfidf | 5 | 17916 | 0.308979 | 0 | FINLAND'S CURRENT ACCOUNT IN DEFICIT IN 1ST QTR |
| Q09 | tfidf | 6 | 15386 | 0.306563 | 1 | SOVIET TRADE DEFICIT WITH WEST SOARS IN 1986 |
| Q09 | tfidf | 7 | 18061 | 0.293315 | 1 | TURKISH TRADE DEFICIT WIDENS IN APRIL |
| Q09 | tfidf | 8 | 18347 | 0.293080 | 1 | N.Z. APRIL CURRENT ACCOUNT DEFICIT NARROWS |
| Q09 | tfidf | 9 | 21392 | 0.288422 | 1 | TURKEY CURRENT ACCOUNT DEFICIT WIDENS IN JULY |
| Q09 | tfidf | 10 | 19377 | 0.288275 | 1 | LATIN, CARIBBEAN NATIONS OPPOSE TRADE BILLS |
| Q09 | cbow | 1 | 20082 | 0.711550 | 0 | Further marked dollar fall would burden U.S. Trade deficit -Stoltenberg |
| Q09 | cbow | 2 | 19569 | 0.645052 | 1 | German May trade surplus 10.6 billion marks (April surplus 8.9 billion) - official |
| Q09 | cbow | 3 | 19062 | 0.619555 | 1 | JAPAN MUST TRY HARDER TO CUT SURPLUS - MINISTER |
| Q09 | cbow | 4 | 17924 | 0.616558 | 1 | UAE TRADE, CURRENT ACCOUNT SURPLUSES NARROW |
| Q09 | cbow | 5 | 17926 | 0.616541 | 1 | UAE TRADE, CURRENT ACCOUNT SURPLUSES NARROW |
| Q09 | cbow | 6 | 17916 | 0.612149 | 0 | FINLAND'S CURRENT ACCOUNT IN DEFICIT IN 1ST QTR |
| Q09 | cbow | 7 | 15386 | 0.601536 | 1 | SOVIET TRADE DEFICIT WITH WEST SOARS IN 1986 |
| Q09 | cbow | 8 | 14862 | 0.593878 | 1 | SOUTH KOREA MOVES TO SLOW GROWTH OF TRADE SURPLUS |
| Q09 | cbow | 9 | 16125 | 0.584009 | 1 | SINGAPORE EXTERNAL TRADE GAINS 8.8 PCT IN QUARTER |
| Q09 | cbow | 10 | 21392 | 0.581884 | 1 | TURKEY CURRENT ACCOUNT DEFICIT WIDENS IN JULY |
| Q09 | skipgram | 1 | 20082 | 0.742324 | 0 | Further marked dollar fall would burden U.S. Trade deficit -Stoltenberg |
| Q09 | skipgram | 2 | 19062 | 0.720822 | 1 | JAPAN MUST TRY HARDER TO CUT SURPLUS - MINISTER |
| Q09 | skipgram | 3 | 17916 | 0.718395 | 0 | FINLAND'S CURRENT ACCOUNT IN DEFICIT IN 1ST QTR |
| Q09 | skipgram | 4 | 15646 | 0.717514 | 1 | TREASURY'S BAKER SAYS TRADE, CURRENT ACCOUNT IMBALANCES "SIMPLY NOT SUSTAINABLE" |
| Q09 | skipgram | 5 | 21392 | 0.715153 | 1 | TURKEY CURRENT ACCOUNT DEFICIT WIDENS IN JULY |
| Q09 | skipgram | 6 | 17924 | 0.715072 | 1 | UAE TRADE, CURRENT ACCOUNT SURPLUSES NARROW |
| Q09 | skipgram | 7 | 14862 | 0.712562 | 1 | SOUTH KOREA MOVES TO SLOW GROWTH OF TRADE SURPLUS |
| Q09 | skipgram | 8 | 17926 | 0.710850 | 1 | UAE TRADE, CURRENT ACCOUNT SURPLUSES NARROW |
| Q09 | skipgram | 9 | 19569 | 0.702560 | 1 | German May trade surplus 10.6 billion marks (April surplus 8.9 billion) - official |
| Q09 | skipgram | 10 | 15386 | 0.701681 | 1 | SOVIET TRADE DEFICIT WITH WEST SOARS IN 1986 |
| Q10 | tfidf | 1 | 19947 | 0.271640 | 1 | PAKISTAN UNVEILS NEW TRADE POLICY TO BOOST EXPORTS |
| Q10 | tfidf | 2 | 19539 | 0.205141 | 1 | BANGLADESH'S TRADE GAP TO WIDEN IN 1987/88 |
| Q10 | tfidf | 3 | 17948 | 0.192921 | 1 | JAPAN CONCERNED AT EC TRADE THREATS - MINISTER |
| Q10 | tfidf | 4 | 20382 | 0.189616 | 0 | IBC SEEKS EXPORT DETAILS TO SET COFFEE QUOTAS |
| Q10 | tfidf | 5 | 17449 | 0.183572 | 1 | SEOUL ANNOUNCES MORE TARIFF CUTS FOR U.S. |
| Q10 | tfidf | 6 | 15648 | 0.180658 | 0 | EC AUTHORISES EXPORT 55,000 TONNES MAIZE,  MAXIMUM REBATE 130 ECUS/TONNE - TRADE SOURCES. |
| Q10 | tfidf | 7 | 18992 | 0.166391 | 1 | U.S.-CANADA TIES SEEN WORSE WITHOUT TRADE PACT |
| Q10 | tfidf | 8 | 15649 | 0.162293 | 0 | EC AUTHORISES EXPORT 65,000 TONNES BARLEY,  MAXIMUM REBATE 138.75 ECUS - BRUSSELS TRADE |
| Q10 | tfidf | 9 | 16094 | 0.159638 | 1 | TAIWAN ANNOUNCES NEW ROUND OF IMPORT TARIFF CUTS |
| Q10 | tfidf | 10 | 16357 | 0.159638 | 1 | TAIWAN ANNOUNCES NEW ROUND OF IMPORT TARIFF CUTS |
| Q10 | cbow | 1 | 19947 | 0.724317 | 1 | PAKISTAN UNVEILS NEW TRADE POLICY TO BOOST EXPORTS |
| Q10 | cbow | 2 | 19539 | 0.705012 | 1 | BANGLADESH'S TRADE GAP TO WIDEN IN 1987/88 |
| Q10 | cbow | 3 | 16627 | 0.671911 | 0 | U.S. TRADE PANEL RULES AGAINST BRAZILIAN  ORANGE JUICE IMPORTS, WILL IMPOSE DUTUES |
| Q10 | cbow | 4 | 20511 | 0.658089 | 1 | EC AGREES TRADE DEAL WITH ARGENTINA |
| Q10 | cbow | 5 | 15798 | 0.658073 | 0 | DISEASE PUTS ZIMBABWE BEEF EXPORTS IN JEOPARDY |
| Q10 | cbow | 6 | 17948 | 0.645564 | 1 | JAPAN CONCERNED AT EC TRADE THREATS - MINISTER |
| Q10 | cbow | 7 | 19835 | 0.644264 | 0 | PAKISTAN ALLOWS PRIVATE COTTON, RICE EXPORTS |
| Q10 | cbow | 8 | 14858 | 0.642909 | 1 | JAPAN MINISTRY SAYS OPEN FARM TRADE WOULD HIT U.S. |
| Q10 | cbow | 9 | 19982 | 0.640287 | 0 | STATE TO CONTROL 70 PCT OF PAKISTAN TEA IMPORTS |
| Q10 | cbow | 10 | 15313 | 0.630387 | 1 | CANADA TO MONITOR STEEL IMPORTS, EXPORTS |
| Q10 | skipgram | 1 | 19947 | 0.720068 | 1 | PAKISTAN UNVEILS NEW TRADE POLICY TO BOOST EXPORTS |
| Q10 | skipgram | 2 | 15262 | 0.717857 | 1 | CANADA PLANS TO MONITOR STEEL IMPORTS, EXPORTS, TRADE MINISTER SAYS |
| Q10 | skipgram | 3 | 19982 | 0.715710 | 0 | STATE TO CONTROL 70 PCT OF PAKISTAN TEA IMPORTS |
| Q10 | skipgram | 4 | 19539 | 0.713040 | 1 | BANGLADESH'S TRADE GAP TO WIDEN IN 1987/88 |
| Q10 | skipgram | 5 | 18352 | 0.703696 | 0 | U.S. MEAT INDUSTRY LAUNCHES CAMPAIGN IN JAPAN |
| Q10 | skipgram | 6 | 16752 | 0.702917 | 0 | TAIWAN LIKELY TO BUY MORE U.S. SOYBEANS |
| Q10 | skipgram | 7 | 14858 | 0.700204 | 1 | JAPAN MINISTRY SAYS OPEN FARM TRADE WOULD HIT U.S. |
| Q10 | skipgram | 8 | 15313 | 0.698542 | 1 | CANADA TO MONITOR STEEL IMPORTS, EXPORTS |
| Q10 | skipgram | 9 | 19835 | 0.698209 | 0 | PAKISTAN ALLOWS PRIVATE COTTON, RICE EXPORTS |
| Q10 | skipgram | 10 | 16225 | 0.690238 | 0 | PAKISTAN CONFIRMS KENYA TEA IMPORT INVESTIGATION |
