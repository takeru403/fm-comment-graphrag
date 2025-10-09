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
from src.data.yfinance_data import YFinanceData
from src.data.news_data_range import get_news_data_range


def _safe_str(value: Any) -> str:
    try:
        return str(value).strip()
    except Exception:
        return ""


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
    #sample = doc_texts[: max(1, max_docs)]
    prompt = (
        "以下のテキスト集合から、知識グラフの三つ組（subject, relation, object）を日本語で抽出してください。\n"
        "- 出力はJSONの配列のみ（余計な文章なし）。\n"
        '- 例: [{"subject":"日経平均","relation":"上昇要因","object":"米国株高"}]\n\n'
        f"【テキスト】\n{'\n---\n'.join(doc_texts)}\n"
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


def build_networkx_from_neo4j(neo4j_graph: Neo4jGraph) -> nx.DiGraph:
    """Neo4j内の (s)-[r]->(o) をNetworkX DiGraphへ展開する。

    - 参照ノイズになりやすい "MENTIONS" 関係は除外
    - ノード属性: label, type
    - エッジ属性: relation, weight
    """
    rows = neo4j_graph.query(
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
        s_key = _safe_str(row.get("s_key", ""))
        o_key = _safe_str(row.get("o_key", ""))
        s_label = _safe_str(row.get("s_label", s_key))
        o_label = _safe_str(row.get("o_label", o_key))
        rel = _safe_str(row.get("rel", ""))
        try:
            weight = float(row.get("weight", 1.0))
        except Exception:
            weight = 1.0
        if not s_key or not o_key:
            continue
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
    return G


def detect_communities(G: nx.DiGraph, max_communities: int = 8) -> List[List[str]]:
    """NetworkXでコミュニティ検出（貪欲モジュラリティ）。

    DiGraphを無向重み付きに変換してから検出する。
    返り値はノードIDのリスト（大きい順に切り出し）。
    """
    if G.number_of_nodes() == 0:
        return []
    UG = G.to_undirected()
    try:
        from networkx.algorithms.community import greedy_modularity_communities
    except Exception:
        return [list(G.nodes())]
    communities = list(greedy_modularity_communities(UG, weight="weight"))
    communities = sorted(communities, key=lambda c: len(c), reverse=True)
    return [list(c) for c in communities[:max_communities]]


def format_community_edges(G: nx.DiGraph, nodes: List[str], max_edges: int = 80) -> str:
    """コミュニティ内の代表エッジを重み順に並べたテキストを返す。"""
    if not nodes:
        return ""
    sub = G.subgraph(nodes).copy()
    edges_sorted = sorted(
        sub.edges(data=True), key=lambda e: float(e[2].get("weight", 1.0)), reverse=True
    )[:max_edges]
    lines: List[str] = []
    for u, v, data in edges_sorted:
        # 表示はGEXFのID（ノードキー）に合わせる
        s_label = _safe_str(u)
        o_label = _safe_str(v)
        rel = _safe_str(data.get("relation", ""))
        if s_label and rel and o_label:
            lines.append(f"{s_label} —{rel}→ {o_label}")
    return "\n".join(lines)


def summarize_community_with_llm(llm: ChatOpenAI, community_edges_text: str) -> str:
    """コミュニティ内のエッジ列から短い日本語要約を生成。"""
    if not community_edges_text.strip():
        return ""
    prompt = (
        "次の関係リストは同一テーマの知識グラフサブグラフです。主要トピック、因果/影響、重要な関係を簡潔に日本語で要約してください。\n"
        "- 箇条書きにせず3〜5文程度。\n\n"
        f"【サブグラフ】\n{community_edges_text}"
    )
    res = llm.invoke([HumanMessage(content=prompt)])
    return _safe_str(getattr(res, "content", ""))


def select_top_communities(
    embedder: OpenAIEmbeddings, query: str, community_texts: List[str], top_k: int = 2
) -> List[int]:
    if not community_texts:
        return []
    try:
        qv = embedder.embed_query(query)
        dvs = embedder.embed_documents(community_texts)
    except Exception:
        return list(range(min(top_k, len(community_texts))))
    import math

    def cos(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb + 1e-12)

    scored = [(i, cos(qv, dv)) for i, dv in enumerate(dvs)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in scored[:top_k]]


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
    upsert_texts_to_neo4j(neo4j_graph=neo4j_graph, llm=llm, texts=top_texts)

    # コミュニティRAG: グラフからコミュニティを検出し、クエリに近いものを選択
    G = build_networkx_from_neo4j(neo4j_graph)
    communities = detect_communities(G, max_communities=8)
    community_edges_texts: List[str] = [
        format_community_edges(G, nodes=c, max_edges=80) for c in communities
    ]
    community_summaries: List[str] = [
        summarize_community_with_llm(llm, t) if t else "" for t in community_edges_texts
    ]
    selectable_texts = [
        (s + "\n" + t).strip() if s else t for s, t in zip(community_summaries, community_edges_texts)
    ]
    sel_idx = select_top_communities(embedder, query, selectable_texts, top_k=4)
    selected_contexts = [selectable_texts[i] for i in sel_idx]
    graph_context = "\n\n---\n\n".join([c for c in selected_contexts if c])

    # Gephi用にGEXFエクスポート
    export_path = project_root / "outputs" / "graph.gexf"
    export_neo4j_to_gexf(neo4j_graph, export_path)

    # 日経平均の終値・差分・変化率の推移（直近最大10ポイント）
    nikkei = YFinanceData("^N225")
    price_trend = nikkei.summarize_closing_trend(max_points=10)
    print(graph_context)
    # 対象期間（news.json の最小/最大 publishedAt）
    start_iso, end_iso = get_news_data_range(project_root / "data" / "news.json")

    user_instruction = (
        "以下のコミュニティ要約（サブグラフ）とニュース要約を根拠に、日経平均連動ファンドの視点から、相場への示唆と運用スタンスを日本語で800文字程度で述べてください。\n"
        "- 箇条書きにせず簡潔に。\n"
        "- 過度な断定を避け、リスク要因も一言触れてください。\n"
        "- 数字や固有名詞は可能な範囲で反映。\n"
        "- 月前半/後半といった表現は使わず、対象期間全体として記述してください。\n\n"
        f"【対象期間】{start_iso} 〜 {end_iso}\n"
        f"【日経平均 終値・差分・変化率の推移】\n{price_trend}\n\n"
        f"【コミュニティ文脈】\n{graph_context}\n\n"
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
