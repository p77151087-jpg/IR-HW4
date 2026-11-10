# Reuters Lab IR HW4

獨立的 Reuters-21578 新聞檢索作業。以 TF-IDF、TF-IDF 加權 CBOW、TF-IDF 加權 Skip-Gram 比較同一批新聞的排名；保留四個工作頁，移除 HW3 的額外排名法、L2 前後教學、PubMed 規則、人工標註、校字與詞空間一致度實驗。

專案與自己的虛擬環境位於 C:\成大專案\IR-HW4，不匯入或依賴 HW3／HW2。原 HW3 成品不修改；沿用內容及原檔指紋見 reports/reuse-provenance.json，完工核對見 reports/hw3-protection-after.json。

## 啟動

雙擊 start_hw4.cmd，瀏覽器會開啟 http://127.0.0.1:8504。啟動工具等待服務就緒後才開網頁；重複啟動會沿用本專案服務，不會停止占用相同 port 的其他程序。停止時雙擊 stop_hw4.cmd，只處理本專案記錄且身分相符的程序。

也可在本專案使用前景模式：

~~~powershell
.\.venv\Scripts\python.exe -B -m streamlit run app.py --server.address 127.0.0.1 --server.port 8504
~~~

本機已備好資料、兩組新模型、完整評估及獨立環境，正常啟動不需要網路，不會自動下載或重訓。8504 與 HW3 的 8503 分開。

### Streamlit Community Cloud

雲端部署使用 `codex/streamlit-cloud` 分支的 `cloud_app.py` 與 Python 3.12。安裝依賴時從 UCI 取得 Reuters-21578, Distribution 1.0，重建語料並核對原始封存檔、語料和模型所綁定的資料指紋。網站讀取已核對的語料與既有模型。雲端參數實驗頁提供已完成的比較結果與 Loss 曲線，自行訓練請使用本機版。

## 四個工作頁

1. **資料概況**：ModApte 訓練／測試篇數、五主題分布、三方法說明、新聞閱讀及完整 TF／DF／IDF 下載。
2. **詞向量分析**從三個角度解讀近鄰詞，分模型列出近鄰名次，並給出語境判讀、同篇共現數及新聞證據。下方並排顯示兩個模型各自的 Top-5，可切換原始 cosine 長條、PCA 或 t-SNE。選擇近鄰可核對 Reuters 原文及訓練 CF／DF。
3. **新聞搜尋與比較**可分別選擇已訓練的 CBOW 與 Skip-Gram，與 TF-IDF 並排看同一查詢的 Top 10。每個原始關鍵字查詢都有模糊句和混用版本，補充題庫見 `resources/query_variants.json`。排名下方的「檢索有改善嗎？」預設顯示目前查詢的指標及相對 TF-IDF 的差值，也可切換到同題三種問法或各類的十題平均。切換模型會沿用目前已提交的查詢，選好問法或輸入新查詢後按「搜尋並比較」才套用。
4. **效果評估**：固定十查詢的平均及逐查詢 Precision@10、MRR／RR、nDCG@10；新增 Q06 改善與局部失敗、Q09 排序退步、Q01 損益方向與主題標籤邊界三個案例。可核對完整名次、cosine、topics、共同查詢詞與新聞全文，下載完整整合報告及證據 ZIP。

## 完整報告與本次內容補充

側欄及詞向量頁可下載整合報告。效果評估頁另提供報告與數據證據 ZIP。報告逐列解讀十詞在雙模型的 100 筆近鄰，列出訓練共現與新聞 ID，並整理三個作業問題。證據包的 analysis.json 收錄去重後 82 組詞對的原文。報告也保留資料切分、計算與訓練流程、十查詢指標、改善與失敗案例及三方法的完整 Top 10。

報告另存於 reports/analysis/runs/ 下的新版本，reports/analysis/current.json 指向完成且已驗證的版本。網站讀取已完成的報告與正式模型。

文字解釋與數據事實分開。詞頻及原文證據只取訓練集，文件案例只取測試集；文件共現不稱作窗口共現，共現為零時明列分開的原文。所有說明是本次結果的分析推論，沒有另建人工相關性標籤，也沒有改寫原 topics、查詢或分數。

原評估 manifest 綁定的計分、資料與評估來源仍全部驗證；其中原 UI 檔案 hash 只代表當時展示版本。這次僅更動展示內容，因此不把 UI 文字變動當成指標過期；新增分析程式、文字資料與衍生檔另有獨立 manifest 與 SHA256，若計分來源或查詢不符仍拒絕使用。

## 資料與公平比較

來源為 [UCI 原始 Reuters-21578](https://kdd.ics.uci.edu/databases/reuters21578/)，以官方 [README 的 ModApte 定義](https://kdd.ics.uci.edu/databases/reuters21578/README.txt) 解析 22 份原始 SGML。

- 完整 21,578 篇；訓練 9,603、測試 3,299、unused 8,676。以 TOPICS=YES 與 LEWISSPLIT=TRAIN/TEST 選取，不因 TOPICS 元素為空而刪除測試新聞。
- 全測試集均為候選，沒有先篩到 earn、acq、crude、grain、trade 才排名。五主題只用來設計查詢及建立相關集合。
- 原檔、下載 archive、官方 README、解析後 JSONL 及 SHA256 保存於 data/reuters-21578。archive SHA256 為 3bae43c9b14e387f76a61b6d82bf98a4fb5d3ef99ef7e7075ff2ccbcf59f9d30。
- 訓練集建立完整 39,012 詞詞表、CF、DF、IDF，測試與查詢只 transform，不新增詞、不重算 IDF。unused 文件不參與任何擬合。
- 文件文字為 TITLE＋BODY，沒有 BODY 的簡訊使用 TITLE；未結構化新聞以 TEXT 回退。日期、ID、topics 不進入向量。空文字若出現保留為零向量。
- NFC、casefold、撇號／連字號正規化、固定 132 停用詞；保留 no／not／without，不 stemming。新聞、查詢及訓練序列使用相同 tokenizer。
- 逐句訓練，共 70,334 句、888,403 tokens；context 不跨文件、標題／正文區塊或偵測句子。逐篇驗證切句後的 token 與全文前處理完全相同。

原始 Reuters 語料、解析後的 JSONL 與使用說明保存在本機的 `data/`。從 GitHub 下載專案後，先執行 `scripts/setup_hw4.ps1` 建立環境並下載、解析語料。腳本會核對重新準備的語料與 `resources/reuters-21578-manifest.json`，讓已發布的模型和評估可以驗證原有資料指紋。

## 計算與模型

TF 是詞在單篇新聞或查詢中的實際次數；CF 是訓練集合總 TF；DF 是含有該詞的訓練新聞篇數。

TF-IDF = TF × log10(N_train / DF_train)。查詢與新聞保留各自 TF，cosine 使用完整詞彙空間的 L2 norm。

加權詞向量 = sum(TF-IDF × 詞向量) / sum(具有詞向量的 TF-IDF)。CBOW／Skip-Gram 各自以相同規則表示新聞與查詢，再以原始 100 維 cosine 排名。OOV 忽略並回報涵蓋率，沒有有效向量的查詢不產生排名。

兩組各以新 Gensim Word2Vec 從頭訓練：100 維、window=5、min_count=2、30 epochs、seed=42、workers=1、negative=5、sample=0.001、ns_exponent=0.75、alpha=0.025、min_alpha=0.0001。只有 sg 架構不同；兩組各 23,172 詞。未使用預訓練向量或舊 HW3 權重。

CBOW 由同一窗口中的上下文預測中心詞；Skip-Gram 由中心詞預測上下文。Gensim 執行訓練核心，本專案負責原始資料準備、fresh initialization、build_vocab／train、negative-sampling 設定、逐 epoch loss、保存及新程序重載驗證。作業說明未指定手刻限制；若課堂另要求手寫優化器，需另補此部分。

候選為全部有效測試向量，保留零分及負 cosine。Embedding 不以共同查詢詞作候選門檻。零向量文件另行計數，不補任意排名；排序僅將分數四捨五入 12 位供穩定 ties，再依數字 NEWID 升冪，保存分數保留原始精度。

詞 Top-5 在各自完整 Word2Vec 詞表搜尋，排除自己及零向量，保留數字和縮寫。預設長條圖以同一尺度呈現正負 cosine。PCA 與 t-SNE 在每張圖使用該模型的中心詞與自己的 Top-5。可選取圖中的任兩詞查看原始 cosine 與完整單位向量距離 √(2 − 2 × cosine)，PCA 另顯示二維投影距離。

t-SNE 以各模型的單位向量及固定 seed 42 獨立投影。Perplexity 隨詞數調整並顯示於圖上。圖上編號代表該模型的近鄰名次，跨模型量化比較使用原始 cosine 或完整向量距離。語境判讀依據 Reuters 訓練新聞，同一詞可回答多題。例如 `oil` 的 `kernel` 是意外近鄰，也有棕櫚油商品情境的原文支持。

## 評估結果與限制

resources/queries.json 在訓練與評估前固定，五主題各兩個英文查詢。相關 iff 目標 topic 出現在測試文件的原始 topics 中，多標籤各自保留。這是資料集主題代理，沒有聲稱已完成針對這十查詢的人工判讀。

| 方法 | Precision@10 | MRR | nDCG@10 |
|---|---:|---:|---:|
| TF-IDF | 0.8600 | 1.0000 | 0.8840 |
| TF-IDF 加權 CBOW | 0.9000 | 0.9000 | 0.8828 |
| TF-IDF 加權 Skip-Gram | 0.9300 | 0.9500 | 0.9266 |

以上均平均 10／10 查詢。Skip-Gram 在本次 P@10 與 nDCG@10 較高，但 TF-IDF 的 MRR 較高；CBOW 的 nDCG@10 也未超過 TF-IDF，不能用一個指標宣稱 embedding 一定較好。

P@10 分母固定 10。每查詢 RR 取完整返回排名的第一篇相關新聞，MRR 為查詢平均。nDCG@10 使用二元 gain 與 log2 折扣。沒有參考正例的查詢才排除並揭露分母；有參考正例但查詢全 OOV 時給零分，不選擇性排除。

目前完整內容請由網站的「下載完整整合報告」取得，實體位置由 reports/analysis/current.json 指向。原 reports/evaluation-v1/REPORT.md 與 reports/WORD_ANALYSIS.md 保留為初版歷史結果。原始數值保存在 metrics_per_query.csv、metrics_summary.csv、rankings.csv（完整排名）、queries.json 與 manifest.json。

限制包含粗粒度 topics、類別不平衡、每主題兩查詢的依賴、單 seed、小型語料與單組參數。加權平均不保留詞序、否定或複雜語意，profit／loss 接近不代表同義；PCA 與 loss 都不是正確率。

## 重現與驗證

到「效果評估 → 參數實驗 → 自行訓練」，在 CBOW、Skip-Gram 兩欄分別設定 `vector_size`、`window`、`min_count`、`epochs`，或展開進階參數調整 `negative`、`sample`、`seed` 和 `workers`。`vector_size` 是詞向量的維度，`window` 是同一句內中心詞與上下文詞的最大距離。可選參考組合填入兩欄，再分開調整。

按「開始訓練」會在本機背景依序訓練兩個模型，頁面顯示目前模型、epoch 和評估進度。每次使用新實驗名稱，留空則自動命名。網站一次啟動一組實驗，會讀取已驗證的訓練句序列與 TF-IDF 權重。完成後可按「比較這次結果」，或到「詞向量分析」選取新模型。模型、評估與工作記錄分別存於 `artifacts/experiments/<名稱>`、`reports/experiments/<名稱>` 和 `reports/training-jobs/<名稱>`。

「比較結果」預設比較 100／150／300 維 × window 2／5／15 × 5／15 epochs 的 18 組參考實驗，固定 min_count 2、negative 5、sample 0.001、seed 42、workers 1。全部使用相同的 ModApte 切分與十個測試查詢。詞向量分析和搜尋的選單都依這 18 組的平均 nDCG@10 分別標記 CBOW 與 Skip-Gram 的最高與最低。搜尋預設選用各自的最高組。網站的模型選單與自行訓練以 100 維為下限。

「詞向量分析」可分別選擇 CBOW 與 Skip-Gram 的參數。參數表下方各用一張圖比較最高、最低與目前選取組合的每輪 Loss。選取最高或最低組合時，圖例會在同一條曲線標示「目前選取」。「參數實驗 → 訓練 Loss 曲線」也可比較最高與最低，或切換「單組實驗」。圖表讀取各 run 的 `epoch_losses.csv`，曲線停在實際訓練的最後一輪。

網站的模型與實驗選單已移除 30 輪的組合。「詞向量分析」會在來源選單下方分欄列出兩個模型的參數。

若從 PowerShell 用同一組設定訓練兩個模型，可使用下列命令。

~~~powershell
Set-Location -LiteralPath 'C:\成大專案\IR-HW4'
.\.venv\Scripts\python.exe -B -X utf8 -m ir_hw4.experiments train my-dim150-w2-e5 --vector-size 150 --window 2 --epochs 5
.\.venv\Scripts\python.exe -B -X utf8 -m ir_hw4.experiments train my-dim150-w5-e15 --vector-size 150 --window 5 --epochs 15
.\.venv\Scripts\python.exe -B -X utf8 -m ir_hw4.experiments train --help
~~~

已完成的實驗可比較十個查詢的平均與逐查詢指標，並在詞向量分析查看各自的 Top-5、PCA 與 t-SNE。語境關係顯示原報告或 `resources/neighbor_context_notes.json` 中已核對的逐詞解讀。其餘近鄰會比較訓練句子中各詞左右兩個詞的用法。共現篇數及新聞原文可在逐詞解讀區核對。

~~~powershell
# 重建獨立環境或補齊已保存的資料；完整時不重訓或覆寫
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_hw4.ps1

# 搜尋
.\.venv\Scripts\python.exe -B -X utf8 -m ir_hw4.cli search "crude oil production" --method skipgram

# 計算與真實四頁 Streamlit AppTest
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\test_hw4.ps1

# 重新解析原始 SGML、獨立 NumPy 分數與保護 HW3 檢查
.\.venv\Scripts\python.exe -B -X utf8 scripts\verify_hw4.py

# 內容補充與原文／排名核對；建立報告會另存新版本，不覆寫舊實驗
.\.venv\Scripts\python.exe -B -X utf8 scripts\build_report.py
.\.venv\Scripts\python.exe -B -X utf8 scripts\verify_analysis.py

# 新實驗請使用新的輸出目錄，不覆寫既有成果
.\.venv\Scripts\python.exe -B -X utf8 -m ir_hw4.cli --run artifacts\another-run train
.\.venv\Scripts\python.exe -B -X utf8 -m ir_hw4.cli --run artifacts\another-run evaluate --output reports\another-evaluation
~~~

換機時使用 Python 3.12，重新建立 .venv 並安裝 requirements-lock.txt，不複製現有虛擬環境。setup 可用 -Python 指定 Python 3.12 執行檔。

初版的 23 項測試記錄保留在 reports/pytest.xml。本次內容補充後共 31 項測試通過，包含十詞逐一切換、三案例切換、來源不符拒絕、報告防損壞及 ZIP 一致性，證據另存 reports/analysis-update-20261105/pytest-final.xml。

另重跑 98,970 個全測試集分數比對、兩組完整模型精確重載及兩次記憶體評估一致性。獨立最大 cosine 誤差小於 6×10^-16、文件加權向量誤差小於 2×10^-15；報告另核對 20 組詞彙證據、26 段原文、三案例的 39 個分數。數值證據見 reports/verification.json，內容證據見 reports/analysis-update-20261105/analysis-verification-final.json。

## 檔案配置

- app.py、ir_hw4/ui.py：四頁介面與快取。
- ir_hw4/dataset.py、text.py、sentences.py：官方 SGML 解析、固定前處理與句界。
- ir_hw4/retrieval.py：train-only TF-IDF、加權詞向量與排名。
- ir_hw4/training.py：明確觸發的新模型訓練、成品保存與指紋檢查。
- ir_hw4/evaluation.py、validation.py：topics 評估及獨立數值核對。
- resources/：固定停用詞及查詢；data/：完整固定語料；artifacts/：完成的模型與向量；reports/：實驗與驗證。
- docs/HW4-assignment.docx：作業說明副本，原始檔未修改。
