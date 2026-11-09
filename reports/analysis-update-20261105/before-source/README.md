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

## 四個工作頁

1. **資料概況**：ModApte 訓練／測試篇數、五主題分布、三方法說明、新聞閱讀及完整 TF／DF／IDF 下載。
2. **詞向量分析**：相同詞的 CBOW／Skip-Gram Top-5、訓練新聞情境、詞對 cosine 與收合的 PCA 圖。PCA 不參與排名。
3. **新聞搜尋與比較**：相同英文查詢的三方法 Top 10、聯集的完整名次對照、新聞全文、查詢涵蓋與 CSV。修改查詢後明示尚未提交，按搜尋才更新結果。
4. **效果評估**：固定十查詢的平均及逐查詢 Precision@10、MRR／RR、nDCG@10；各查詢 Top 10 的主題標籤、方法差值、CSV 及完整報告下載。

## 資料與公平比較

來源為 [UCI 原始 Reuters-21578](https://kdd.ics.uci.edu/databases/reuters21578/)，以官方 [README 的 ModApte 定義](https://kdd.ics.uci.edu/databases/reuters21578/README.txt) 解析 22 份原始 SGML。

- 完整 21,578 篇；訓練 9,603、測試 3,299、unused 8,676。以 TOPICS=YES 與 LEWISSPLIT=TRAIN/TEST 選取，不因 TOPICS 元素為空而刪除測試新聞。
- 全測試集均為候選，沒有先篩到 earn、acq、crude、grain、trade 才排名。五主題只用來設計查詢及建立相關集合。
- 原檔、下載 archive、官方 README、解析後 JSONL 及 SHA256 保存於 data/reuters-21578。archive SHA256 為 3bae43c9b14e387f76a61b6d82bf98a4fb5d3ef99ef7e7075ff2ccbcf59f9d30。
- 訓練集建立完整 39,012 詞詞表、CF、DF、IDF，測試與查詢只 transform，不新增詞、不重算 IDF。unused 文件不參與任何擬合。
- 文件文字為 TITLE＋BODY，沒有 BODY 的簡訊使用 TITLE；未結構化新聞以 TEXT 回退。日期、ID、topics 不進入向量。空文字若出現保留為零向量。
- NFC、casefold、撇號／連字號正規化、固定 132 停用詞；保留 no／not／without，不 stemming。新聞、查詢及訓練序列使用相同 tokenizer。
- 逐句訓練，共 70,334 句、888,403 tokens；context 不跨文件、標題／正文區塊或偵測句子。逐篇驗證切句後的 token 與全文前處理完全相同。

原始 Reuters README 的資料使用說明保存在資料目錄；本專案作本機課程研究用途，未發布新聞資料至外部服務。

## 計算與模型

TF 是詞在單篇新聞或查詢中的實際次數；CF 是訓練集合總 TF；DF 是含有該詞的訓練新聞篇數。

TF-IDF = TF × log10(N_train / DF_train)。查詢與新聞保留各自 TF，cosine 使用完整詞彙空間的 L2 norm。

加權詞向量 = sum(TF-IDF × 詞向量) / sum(具有詞向量的 TF-IDF)。CBOW／Skip-Gram 各自以相同規則表示新聞與查詢，再以原始 100 維 cosine 排名。OOV 忽略並回報涵蓋率，沒有有效向量的查詢不產生排名。

兩組各以新 Gensim Word2Vec 從頭訓練：100 維、window=5、min_count=2、30 epochs、seed=42、workers=1、negative=5、sample=0.001、ns_exponent=0.75、alpha=0.025、min_alpha=0.0001。只有 sg 架構不同；兩組各 23,172 詞。未使用預訓練向量或舊 HW3 權重。

CBOW 由同一窗口中的上下文預測中心詞；Skip-Gram 由中心詞預測上下文。Gensim 執行訓練核心，本專案負責原始資料準備、fresh initialization、build_vocab／train、negative-sampling 設定、逐 epoch loss、保存及新程序重載驗證。作業說明未指定手刻限制；若課堂另要求手寫優化器，需另補此部分。

候選為全部有效測試向量，保留零分及負 cosine。Embedding 不以共同查詢詞作候選門檻。零向量文件另行計數，不補任意排名；排序僅將分數四捨五入 12 位供穩定 ties，再依數字 NEWID 升冪，保存分數保留原始精度。

詞 Top-5 在各自完整 Word2Vec 詞表搜尋，排除自己及零向量，沒有為了美化結果事後移除數字或縮寫。cosine 為 signed；負分不截成零。PCA 只投影畫面選詞的 L2 詞向量，座標不參與近鄰或檢索。

## 評估結果與限制

resources/queries.json 在訓練與評估前固定，五主題各兩個英文查詢。相關 iff 目標 topic 出現在測試文件的原始 topics 中，多標籤各自保留。這是資料集主題代理，沒有聲稱已完成針對這十查詢的人工判讀。

| 方法 | Precision@10 | MRR | nDCG@10 |
|---|---:|---:|---:|
| TF-IDF | 0.8600 | 1.0000 | 0.8840 |
| TF-IDF 加權 CBOW | 0.9000 | 0.9000 | 0.8828 |
| TF-IDF 加權 Skip-Gram | 0.9300 | 0.9500 | 0.9266 |

以上均平均 10／10 查詢。Skip-Gram 在本次 P@10 與 nDCG@10 較高，但 TF-IDF 的 MRR 較高；CBOW 的 nDCG@10 也未超過 TF-IDF，不能用一個指標宣稱 embedding 一定較好。

P@10 分母固定 10。每查詢 RR 取完整返回排名的第一篇相關新聞，MRR 為查詢平均。nDCG@10 使用二元 gain 與 log2 折扣。沒有參考正例的查詢才排除並揭露分母；有參考正例但查詢全 OOV 時給零分，不選擇性排除。

完整結果與實際排名例子見 reports/evaluation-v1/REPORT.md，詞近鄰觀察見 reports/WORD_ANALYSIS.md。原始數值保存在 metrics_per_query.csv、metrics_summary.csv、rankings.csv（完整排名）、queries.json 與 manifest.json。

限制包含粗粒度 topics、類別不平衡、每主題兩查詢的依賴、單 seed、小型語料與單組參數。加權平均不保留詞序、否定或複雜語意，profit／loss 接近不代表同義；PCA 與 loss 都不是正確率。

## 重現與驗證

~~~powershell
# 重建獨立環境或補齊已保存的資料；完整時不重訓或覆寫
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_hw4.ps1

# 搜尋
.\.venv\Scripts\python.exe -B -X utf8 -m ir_hw4.cli search "crude oil production" --method skipgram

# 計算與真實四頁 Streamlit AppTest
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\test_hw4.ps1

# 重新解析原始 SGML、獨立 NumPy 分數與保護 HW3 檢查
.\.venv\Scripts\python.exe -B -X utf8 scripts\verify_hw4.py

# 新實驗請使用新的輸出目錄，不覆寫既有成果
.\.venv\Scripts\python.exe -B -X utf8 -m ir_hw4.cli --run artifacts\another-run train
.\.venv\Scripts\python.exe -B -X utf8 -m ir_hw4.cli --run artifacts\another-run evaluate --output reports\another-evaluation
~~~

換機時使用 Python 3.12，重新建立 .venv 並安裝 requirements-lock.txt，不複製現有虛擬環境。setup 可用 -Python 指定 Python 3.12 執行檔。

驗證包含 23 個測試、98,970 個全測試集分數比對、兩組完整模型精確重載，以及兩次記憶體評估完全一致。獨立最大 cosine 誤差小於 6×10^-16、文件加權向量誤差小於 2×10^-15。證據見 reports/pytest.xml 及 reports/verification.json。

## 檔案配置

- app.py、ir_hw4/ui.py：四頁介面與快取。
- ir_hw4/dataset.py、text.py、sentences.py：官方 SGML 解析、固定前處理與句界。
- ir_hw4/retrieval.py：train-only TF-IDF、加權詞向量與排名。
- ir_hw4/training.py：明確觸發的新模型訓練、成品保存與指紋檢查。
- ir_hw4/evaluation.py、validation.py：topics 評估及獨立數值核對。
- resources/：固定停用詞及查詢；data/：完整固定語料；artifacts/：完成的模型與向量；reports/：實驗與驗證。
- docs/HW4-assignment.docx：作業說明副本，原始檔未修改。
