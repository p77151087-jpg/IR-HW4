# IR HW4 工作界線

- 專案位於 C:\成大專案\IR-HW4。HW3、HW2 為原始成品，保持唯讀，不將它們加入 sys.path，不依賴它們的環境或活動資料。
- 使用本專案 .venv\Scripts\python.exe 執行與測試。不要複製其他專案的虛擬環境。
- 結構探索先查 codebase-memory project/generation，再 search_graph、trace_path、get_code_snippet。預設 Tier 2；對每個依據檔案查 check_index_coverage。新碼或缺口直接讀原碼。
- data、artifacts、reports 刻意不進 graph；依檔案、manifest、SHA256 核對，不將圖中無結果視為不存在。
- Reuters 使用原始 SGML 的 ModApte 屬性切分。TOPICS=YES 不代表 TOPICS 元素非空。保留完整測試候選及多標籤，不只搜尋五個目標主題。
- 詞表、DF、IDF、Word2Vec 僅以訓練集擬合。新聞與查詢共用前處理及 weighting；topics 不得進入文字向量。
- topics 是資料集主題代理，不是針對查詢的人工判讀。詞近鄰、PCA、loss 不得當成檢索正確率。
- Embedding 不能沿用 lexical 候選限制。保留 signed cosine，揭露 OOV、零向量與有效候選數。
- 網站唯讀使用既有模型，僅「參數實驗」明確按下「開始訓練」時，可用既有訓練集啟動本機背景程序。新訓練與評估使用新的輸出目錄，拒絕覆寫既有 run。網站不下載資料。
- 修改計分或評估後執行 scripts/test_hw4.ps1 與 scripts/verify_hw4.py。模型來源變更須另建新 run；介面應拒絕使用來源或資料指紋不符的成品。
