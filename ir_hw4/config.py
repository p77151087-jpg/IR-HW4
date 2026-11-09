from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "reuters-21578"
RUN = ROOT / "artifacts" / "reuters-modapte-v1"
TOPICS = {"earn": "企業獲利", "acq": "企業收購", "crude": "原油", "grain": "穀物", "trade": "國際貿易"}
METHODS = {"tfidf": "TF-IDF", "cbow": "TF-IDF 加權 CBOW", "skipgram": "TF-IDF 加權 Skip-Gram"}
ANALYSIS_WORDS = ["profit", "loss", "earnings", "acquisition", "merger", "oil", "petroleum", "wheat", "grain", "trade"]
DATA_URL = "https://kdd.ics.uci.edu/databases/reuters21578/reuters21578.tar.gz"
README_URL = "https://kdd.ics.uci.edu/databases/reuters21578/README.txt"
