from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config.setting import Settings
from src.common.load_news_documents import load_news_documents
from src.common.rag_common import retrieve_top_k
from src.common.read_prompt import read_prompt
from src.common.rag_common import retrieve_top_k
from src.data.yfinance_data import YFinanceData

def build_context(snippets: List[Tuple[str, float]], max_chars: int = 2400) -> str:
    lines: List[str] = []
    for doc_text, score in snippets:
        lines.append(doc_text)
        lines.append("---")

    context = "\n".join(lines)
    if len(context) > max_chars:
        context = context[:max_chars]
    return context


def generate_comment() -> str:
    # パス解決
    here = Path(__file__).resolve()
    project_root = here.parents[2]
    prompt_path = project_root / "src" / "common" / "prompt.yml"

    # 設定とクライアント
    settings = Settings()
    llm = ChatOpenAI(
        model="gpt-4o-mini", temperature=0, api_key=settings.OPENAI_API_KEY
    )
    embedder = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=settings.OPENAI_API_KEY
    )

    def _export_embeddings_for_projector(
        texts: List[str],
        embedder: OpenAIEmbeddings,
        out_dir: Path,
        filename_prefix: str = "embeddings",
    ) -> None:
        """Embedding Projectorで読み込めるTSVを出力する。

        出力: out_dir/{prefix}_metadata.tsv, out_dir/{prefix}_vectors.tsv
        - metadata.tsv: 1列目にテキスト（改行は空白に置換、タブはスペースに）
        - vectors.tsv: 各ベクトル要素をタブ区切り
        """
        if not texts:
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        vectors = embedder.embed_documents(texts)
        meta_path = out_dir / f"{filename_prefix}_metadata.tsv"
        vec_path = out_dir / f"{filename_prefix}_vectors.tsv"

        # メタデータ（ラベルのみ）: 先頭行ベースで120文字に短縮、インデックス付与
        with open(meta_path, "w", encoding="utf-8") as f_meta:
            for i, t in enumerate(texts):
                first_line = (t.splitlines()[0] if t else "").strip()
                base_label = first_line if first_line else t
                if not base_label:
                    base_label = f"doc-{i}"
                sanitized = base_label.replace("\t", " ").replace("\n", " ")
                short = sanitized[:120]
                label = f"{i:03d}: {short}"
                f_meta.write(f"{label}\n")

        # ベクトル
        with open(vec_path, "w", encoding="utf-8") as f_vec:
            for vec in vectors:
                f_vec.write("\t".join(str(x) for x in vec) + "\n")

    # 入力データ（JSON/MD 等の差異をこのファイル内で吸収）
    raw_docs: Any = load_news_documents(news_path="data/news.json")
    documents: List[str] = []
    # dict（NewsAPI形式）とエラー応答の考慮
    if isinstance(raw_docs, dict):
        if "articles" in raw_docs:
            for art in raw_docs.get("articles", []):
                if not isinstance(art, dict):
                    continue
                title = (art.get("title") or "").strip()
                desc = (art.get("description") or "").strip()
                content = (art.get("content") or "").strip()
                text = "\n".join([p for p in [title, desc, content] if p])
                if text:
                    documents.append(text)
        else:
            status = str(raw_docs.get("status") or "").lower()
            message = str(raw_docs.get("message") or "").strip()
            if status and status != "ok":
                raise ValueError(
                    f"news.json にエラー応答が保存されています: status={status}, message={message}"
                )
    # list
    elif isinstance(raw_docs, list):
        for item in raw_docs:
            if isinstance(item, str):
                if item.strip():
                    documents.append(item)
            elif isinstance(item, dict):
                title = (item.get("title") or "").strip()
                desc = (item.get("description") or "").strip()
                content = (item.get("content") or "").strip()
                page_content = (item.get("page_content") or "").strip()
                text = "\n".join([p for p in [title, desc, content, page_content] if p])
                if text:
                    documents.append(text)
    # 単一文字列
    elif isinstance(raw_docs, str):
        if raw_docs.strip():
            documents = [raw_docs]
    else:
        documents = [str(raw_docs)]

    prompt = read_prompt(prompt_path)

    # 想定クエリ（N225運用に関連するトピックを広くカバー）
    query = "日経平均 株価 東京市場 半導体 金利 為替 米国株 景気 FRB インフレ 決算"
    top_docs = retrieve_top_k(query=query, documents=documents, embedder=embedder, k=8)
    context = build_context(top_docs)

    out_dir = (Path(__file__).resolve().parents[2] / "outputs" / "embedding_projector")
    _export_embeddings_for_projector(
        texts=documents,
        embedder=embedder,
        out_dir=out_dir,
        filename_prefix="news_docs",
    )
    # 日経平均の終値・差分・変化率の推移（直近最大10ポイント）
    nikkei = YFinanceData("^N225")
    price_trend = nikkei.summarize_closing_trend(max_points=10)
    print(price_trend)
    user_instruction = (
        "以下のニュース要約を根拠に、日経平均連動ファンドの視点から、相場への示唆と運用スタンスを日本語で400文字程度で述べてください。\n"
        "- 箇条書きにせず簡潔に。\n"
        "- 過度な断定を避け、リスク要因も一言触れてください。\n"
        "- 数字や固有名詞は可能な範囲で反映。\n\n"
        f"【日経平均 終値・差分・変化率の推移】\n{price_trend}\n\n"
        f"【ニュース要約】\n{context}"
    )
    res = llm.invoke(
        [SystemMessage(content=prompt["system_prompt"]), HumanMessage(content=user_instruction)]
    )
    output = (res.content or "").strip()
    return output


def main() -> None:
    comment = generate_comment()
    print(comment)


if __name__ == "__main__":
    main()
