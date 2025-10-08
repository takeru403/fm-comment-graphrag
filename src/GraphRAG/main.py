from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_experimental.graph_transformers import LLMGraphTransformer

# from langchain_community.graphs import Neo4jGraph
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config.setting import Settings
from src.common.load_news_documents import load_news_documents
from src.common.rag_common import retrieve_top_k
from src.common.read_prompt import read_prompt


def _normalize_to_texts(raw_docs: Any) -> List[str]:
    texts: List[str] = []
    if isinstance(raw_docs, dict) and "articles" in raw_docs:
        for art in raw_docs.get("articles", []):
            if not isinstance(art, dict):
                continue
            title = (art.get("title") or "").strip()
            desc = (art.get("description") or "").strip()
            content = (art.get("content") or "").strip()
            text = "\n".join([p for p in [title, desc, content] if p])
            if text:
                texts.append(text)
                print(text)
        return texts
    if isinstance(raw_docs, list):
        for item in raw_docs:
            if isinstance(item, str):
                if item.strip():
                    texts.append(item)
            elif isinstance(item, dict):
                title = (item.get("title") or "").strip()
                desc = (item.get("description") or "").strip()
                content = (item.get("content") or "").strip()
                page_content = (item.get("page_content") or "").strip()
                text = "\n".join([p for p in [title, desc, content, page_content] if p])
                if text:
                    texts.append(text)
        return texts
    if isinstance(raw_docs, str):
        return [raw_docs] if raw_docs.strip() else []
    return [str(raw_docs)]


def extract_triples_for_docs(
    llm: ChatOpenAI, doc_texts: List[str], max_docs: int = 8
) -> List[Dict[str, str]]:
    triples: List[Dict[str, str]] = []
    sample = doc_texts[: max(1, max_docs)]
    prompt = (
        "以下のテキスト集合から、知識グラフの三つ組（subject, relation, object）を日本語で抽出してください。\n"
        "- 出力はJSONの配列のみ（余計な文章なし）。\n"
        '- 例: [{"subject":"日経平均","relation":"上昇要因","object":"米国株高"}]\n\n'
        f"【テキスト】\n{'\n---\n'.join(sample)}\n"
    )
    res = llm.invoke([HumanMessage(content=prompt)])
    content = (res.content or "[]").strip()
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                s = str(item.get("subject", "")).strip()
                r = str(item.get("relation", "")).strip()
                o = str(item.get("object", "")).strip()
                if s and r and o:
                    triples.append({"subject": s, "relation": r, "object": o})
    except Exception:
        pass
    return triples


def build_graph_context(triples: List[Dict[str, str]], max_lines: int = 80) -> str:
    lines: List[str] = []
    for t in triples[:max_lines]:
        s = t.get("subject", "")
        r = t.get("relation", "")
        o = t.get("object", "")
        if s and r and o:
            lines.append(f"{s} —{r}→ {o}")
    return "\n".join(lines)


def upsert_texts_to_neo4j(
    neo4j_graph: Neo4jGraph, llm: ChatOpenAI, texts: List[str]
) -> None:
    transformer = LLMGraphTransformer(llm=llm)
    doc_objs = [
        Document(page_content=t) for t in texts if isinstance(t, str) and t.strip()
    ]
    if not doc_objs:
        return
    graph_docs = transformer.convert_to_graph_documents(doc_objs)
    neo4j_graph.add_graph_documents(graph_docs, include_source=True)


def export_neo4j_to_gexf(neo4j_graph: Neo4jGraph, export_path: Path) -> None:
    # ノード/リレーションをCypherで取得し、NetworkXに展開
    rows = neo4j_graph.query(
        # cyperでクエリ作成
        """
        MATCH (s)-[r]->(o)
        RETURN
          coalesce(s.id, elementId(s), toString(id(s))) as s_key,
          coalesce(s.name, s.title, s.label, head(labels(s)), elementId(s)) as s_label,
          coalesce(o.id, elementId(o), toString(id(o))) as o_key,
          coalesce(o.name, o.title, o.label, head(labels(o)), elementId(o)) as o_label,
          type(r) as rel,
          coalesce(r.weight, 1.0) as weight
        """
    )
    G = nx.DiGraph()
    for row in rows:
        s_key = str(row.get("s_key", "")).strip()
        o_key = str(row.get("o_key", "")).strip()
        s_label = str(row.get("s_label", s_key)).strip()
        o_label = str(row.get("o_label", o_key)).strip()
        rel = str(row.get("rel", "")).strip()
        weight = float(row.get("weight", 1.0))
        if not s_key or not o_key:
            continue
        # ノイズになりやすい参照関係は除外（ハッシュ由来の文書ノードを避ける）
        if rel.upper() == "MENTIONS":
            continue
        if not G.has_node(s_key):
            G.add_node(s_key, label=s_label or s_key, type="entity")
        if not G.has_node(o_key):
            G.add_node(o_key, label=o_label or o_key, type="entity")
        if G.has_edge(s_key, o_key):
            G[s_key][o_key]["weight"] = G[s_key][o_key].get("weight", 0) + weight
        else:
            G.add_edge(s_key, o_key, relation=rel, weight=weight)

    export_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(G, export_path)


def generate_comment() -> str:
    here = Path(__file__).resolve()
    project_root = here.parents[2]
    prompt_path = project_root / "src" / "common" / "prompt.yml"

    settings = Settings()
    llm = ChatOpenAI(
        model="gpt-4o-mini", temperature=0, api_key=settings.OPENAI_API_KEY
    )
    embedder = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=settings.OPENAI_API_KEY
    )

    # Neo4j 接続
    neo4j_uri = settings.NEO4J_URI
    neo4j_user = settings.NEO4J_USERNAME
    neo4j_pass = settings.NEO4J_PASSWORD
    if not (neo4j_uri and neo4j_user and neo4j_pass):
        raise EnvironmentError(
            "NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD を設定してください"
        )
    neo4j_graph = Neo4jGraph(url=neo4j_uri, username=neo4j_user, password=neo4j_pass)

    raw_docs: Any = load_news_documents(news_path=project_root / "data" / "news.json")
    doc_texts = _normalize_to_texts(raw_docs)

    sp = read_prompt(prompt_path)
    system_prompt = sp.get("system_prompt") if isinstance(sp, dict) else str(sp)
    # ここのqueryは要検討。
    query = "日経平均 株価 東京市場 半導体 金利 為替 米国株 景気 FRB インフレ 決算"
    top_docs = retrieve_top_k(query=query, documents=doc_texts, embedder=embedder, k=8)
    top_texts = [t for t, _ in top_docs]

    # LLMでグラフ抽出→Neo4jへアップサート
    upsert_texts_to_neo4j(neo4j_graph=neo4j_graph, llm=llm, texts=top_texts[:3])
    # コメント用の簡易文脈は従来の三つ組抽出を活用
    triples = extract_triples_for_docs(llm=llm, doc_texts=top_texts, max_docs=3)
    graph_context = build_graph_context(triples)

    # Gephi用にGEXFエクスポート
    export_path = project_root / "outputs" / "graph.gexf"
    export_neo4j_to_gexf(neo4j_graph, export_path)

    user_instruction = (
        "以下の知識グラフ要約（トリプル）とニュース要約を根拠に、日経平均連動ファンドの視点から、相場への示唆と運用スタンスを日本語で400文字程度で述べてください。\n"
        "- 箇条書きにせず簡潔に。\n"
        "- 過度な断定を避け、リスク要因も一言触れてください。\n"
        "- 数字や固有名詞は可能な範囲で反映。\n\n"
        f"【知識グラフ要約】\n{graph_context}\n\n"
        f"【ニュース要約】\n{'\n---\n'.join(top_texts)}"
    )

    res = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_instruction)]
    )
    output = (res.content or "").strip()
    return output


def main() -> None:
    comment = generate_comment()
    print(comment)


if __name__ == "__main__":
    main()
