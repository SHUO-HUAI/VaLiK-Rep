import os
import re
import json
import glob
import math
import argparse
import subprocess
import asyncio
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any

import requests
import numpy as np


try:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.ollama import ollama_model_complete
    from lightrag.utils import EmbeddingFunc
except Exception:
    LightRAG = None
    QueryParam = None
    ollama_model_complete = None
    EmbeddingFunc = None


STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "by", "is", "are",
    "was", "were", "be", "been", "being", "as", "at", "from", "that", "this", "these", "those",
    "it", "its", "into", "than", "then", "which", "what", "when", "where", "why", "how", "do",
    "does", "did", "can", "could", "would", "should", "about", "above", "below", "under", "over",
    "between", "during", "through", "because", "if", "while", "after", "before", "question", "choices"
}


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_cmd(cmd: List[str], cwd: Optional[str] = None):
    print("\n[RUN]", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def ask_ollama(model: str, prompt: str, host: str = "http://localhost:11434", temperature: float = 0.0) -> str:
    url = f"{host}/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a science question answering assistant. "
                    "Answer multiple-choice questions carefully. "
                    "Return your answer in strict JSON with keys: answer, reason."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"]


def extract_json_answer(text: str) -> Tuple[Optional[str], str]:
    text = text.strip()

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            obj = json.loads(candidate)
            ans = str(obj.get("answer", "")).strip().upper()
            reason = str(obj.get("reason", "")).strip()
            if ans:
                return ans, reason
        except Exception:
            pass

    match = re.search(r"\b([A-E])\b", text.upper())
    if match:
        return match.group(1), text

    return None, text


def answer_index_to_letter(idx: int) -> str:
    return chr(ord("A") + idx)


def build_question_prompt(
    question: str,
    choices: List[str],
    knowledge: str = "",
    mode: str = "baseline",
) -> str:
    choice_lines = []
    for i, c in enumerate(choices):
        choice_lines.append(f"{answer_index_to_letter(i)}. {c}")
    choice_block = "\n".join(choice_lines)

    if mode == "rag":
        knowledge_header = "Retrieved visual description snippets (text-only RAG baseline)"
    elif mode == "kg":
        knowledge_header = "Retrieved graph-grounded visual knowledge (KG / LightRAG)"
    else:
        knowledge_header = ""

    if mode in {"rag", "kg"} and knowledge.strip():
        prompt = f"""
Answer the following ScienceQA multiple-choice question.

Question:
{question}

Choices:
{choice_block}

{knowledge_header}:
{knowledge}

Please choose the single best answer.
Return strict JSON only, for example:
{{"answer":"A","reason":"brief explanation"}}
""".strip()
    else:
        prompt = f"""
Answer the following ScienceQA multiple-choice question.

Question:
{question}

Choices:
{choice_block}

Please choose the single best answer.
Return strict JSON only, for example:
{{"answer":"A","reason":"brief explanation"}}
""".strip()

    return prompt


def find_image_files(image_dir: str) -> List[str]:
    exts = ["*.png", "*.jpg", "*.jpeg"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(image_dir, ext)))
    return sorted(files)


def stage1_txt_path(image_path: str) -> str:
    base = os.path.splitext(image_path)[0]
    return base + ".blip2-flan-t5.txt"


def stage2_txt_path(image_path: str) -> str:
    base = os.path.splitext(image_path)[0]
    return base + ".blip2-flan-t5.qwen2vl2b.txt"


def final_coe_txt_path(image_path: str) -> str:
    base = os.path.splitext(image_path)[0]
    return base + ".blip2-flan-t5.qwen2vl2b.llava7b.txt"


def final_filtered_txt_path(image_path: str) -> str:
    base = os.path.splitext(image_path)[0]
    return base + ".blip2-flan-t5.qwen2vl2b.llava7b_filtered.txt"


def ensure_coe_and_prune_for_image(
    image_path: str,
    repo_root: str,
    threshold: float = 0.20,
    mode: str = "sentence",
):
    image_dir = os.path.dirname(image_path)
    coe_final_txt = final_coe_txt_path(image_path)
    filtered_txt = final_filtered_txt_path(image_path)

    if os.path.exists(filtered_txt):
        return

    if not os.path.exists(coe_final_txt):
        run_cmd(
            [
                "python",
                "src/CoE_Image_to_Text.py",
                "--input",
                image_dir,
                "blip2",
                "--blip2_version",
                "flan-t5",
            ],
            cwd=repo_root,
        )

        run_cmd(
            [
                "python",
                "src/CoE_Image_to_Text.py",
                "--input",
                image_dir,
                "--previous_prefixes",
                "blip2-flan-t5",
                "qwen2-vl",
                "--qwen2vl_version",
                "2b",
            ],
            cwd=repo_root,
        )

        run_cmd(
            [
                "python",
                "src/CoE_Image_to_Text.py",
                "--input",
                image_dir,
                "--previous_prefixes",
                "blip2-flan-t5,qwen2vl2b",
                "llava",
                "--llava_version",
                "7b",
            ],
            cwd=repo_root,
        )

    if not os.path.exists(filtered_txt):
        run_cmd(
            [
                "python",
                "src/Prune/similarity_verification.py",
                "--image_path",
                image_path,
                "--text_path",
                coe_final_txt,
                "--threshold",
                str(threshold),
                "--mode",
                mode,
            ],
            cwd=repo_root,
        )

    if not os.path.exists(filtered_txt):
        raise FileNotFoundError(f"Filtered txt not generated: {filtered_txt}")


def read_text_if_exists(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def normalize_tokens(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def split_text_into_chunks(text: str, chunk_by: str = "sentence") -> List[str]:
    text = text.strip()
    if not text:
        return []

    if chunk_by == "paragraph":
        chunks = [x.strip() for x in re.split(r"\n\s*\n", text) if x.strip()]
        return chunks

    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def build_rag_corpus_for_image(image_path: str, rag_source: str = "all") -> List[Tuple[str, str]]:
    candidates: List[Tuple[str, str]] = []

    source_to_path = {
        "blip2": [("blip2", stage1_txt_path(image_path))],
        "coe": [("coe_final", final_coe_txt_path(image_path))],
        "filtered": [("filtered", final_filtered_txt_path(image_path))],
        "all": [
            ("blip2", stage1_txt_path(image_path)),
            ("qwen2vl", stage2_txt_path(image_path)),
            ("coe_final", final_coe_txt_path(image_path)),
            ("filtered", final_filtered_txt_path(image_path)),
        ],
    }

    for source_name, path in source_to_path[rag_source]:
        text = read_text_if_exists(path)
        if text:
            candidates.append((source_name, text))

    return candidates


def rank_chunks(query: str, chunks: List[str], topk: int) -> List[Tuple[str, float]]:
    if not chunks:
        return []

    query_tokens = normalize_tokens(query)
    if not query_tokens:
        return [(c, 0.0) for c in chunks[:topk]]

    q_count = Counter(query_tokens)
    query_vocab = set(query_tokens)
    doc_freq = Counter()

    chunk_tokenized = []
    for chunk in chunks:
        toks = normalize_tokens(chunk)
        chunk_tokenized.append(toks)
        for tok in set(toks):
            doc_freq[tok] += 1

    n_docs = len(chunks)
    scored: List[Tuple[str, float]] = []
    for chunk, toks in zip(chunks, chunk_tokenized):
        if not toks:
            scored.append((chunk, 0.0))
            continue

        t_count = Counter(toks)
        overlap = len(query_vocab.intersection(set(toks)))
        tfidf = 0.0
        for tok, qv in q_count.items():
            if tok not in t_count:
                continue
            idf = math.log((1 + n_docs) / (1 + doc_freq[tok])) + 1.0
            tfidf += qv * t_count[tok] * idf

        length_penalty = 1.0 / (1.0 + 0.02 * max(len(toks) - 30, 0))
        score = (tfidf + 0.5 * overlap) * length_penalty
        scored.append((chunk, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:topk]


def collect_rag_knowledge_for_problem(
    problem_id: str,
    problem_info: Dict,
    split: str,
    scienceqa_root: str,
    repo_root: str,
    threshold: float = 0.20,
    prune_mode: str = "sentence",
    auto_build: bool = True,
    rag_source: str = "all",
    rag_chunk_by: str = "sentence",
    rag_topk: int = 5,
) -> Tuple[str, List[Dict]]:
    image_dir = os.path.join(scienceqa_root, "images", split, str(problem_id))
    if not os.path.isdir(image_dir):
        return "", []

    image_files = find_image_files(image_dir)
    if not image_files:
        return "", []

    query = problem_info["question"] + "\n" + "\n".join(
        f"{answer_index_to_letter(i)}. {c}" for i, c in enumerate(problem_info["choices"])
    )

    retrieved_blocks: List[str] = []
    retrieval_debug: List[Dict] = []

    for image_path in image_files:
        if auto_build:
            ensure_coe_and_prune_for_image(
                image_path=image_path,
                repo_root=repo_root,
                threshold=threshold,
                mode=prune_mode,
            )

        corpus_docs = build_rag_corpus_for_image(image_path, rag_source=rag_source)
        chunks_with_meta: List[Tuple[str, str]] = []
        for source_name, text in corpus_docs:
            for chunk in split_text_into_chunks(text, chunk_by=rag_chunk_by):
                chunks_with_meta.append((source_name, chunk))

        ranked = rank_chunks(query, [c for _, c in chunks_with_meta], topk=rag_topk)
        ranked_set = {c for c, _ in ranked}

        selected = []
        for source_name, chunk in chunks_with_meta:
            if chunk in ranked_set:
                score = next(score for text, score in ranked if text == chunk)
                selected.append((source_name, chunk, score))

        selected.sort(key=lambda x: x[2], reverse=True)
        if selected:
            retrieved_blocks.append("\n".join([f"- {chunk}" for _, chunk, _ in selected]))
            retrieval_debug.append(
                {
                    "image_path": image_path,
                    "selected": [
                        {"source": src, "score": round(score, 4), "text": chunk}
                        for src, chunk, score in selected
                    ],
                }
            )

    return "\n\n".join(retrieved_blocks).strip(), retrieval_debug


def make_problem_kg_working_dir(base_dir: str, split: str, problem_id: str) -> str:
    return os.path.join(base_dir, f"{split}_{problem_id}")


def _normalize_ollama_embed_response(data: Dict[str, Any]) -> np.ndarray:
    if "embeddings" not in data:
        raise ValueError(f"Ollama embed response missing 'embeddings': {data}")

    emb = data["embeddings"]
    if not emb:
        return np.zeros((0, 0), dtype=np.float32)

    if isinstance(emb[0], list):
        arr = np.asarray(emb, dtype=np.float32)
    elif isinstance(emb[0], (int, float)):
        arr = np.asarray([emb], dtype=np.float32)
    else:
        raise ValueError(f"Unexpected embedding format: {type(emb)}")

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D embedding array, got shape {arr.shape}")

    return arr


async def custom_ollama_embed(texts: List[str], embed_model: str, host: str) -> np.ndarray:
    if isinstance(texts, str):
        texts = [texts]

    payload = {
        "model": embed_model,
        "input": texts,
    }

    def _post():
        url = f"{host}/api/embed"
        r = requests.post(url, json=payload, timeout=600)
        r.raise_for_status()
        return r.json()

    data = await asyncio.to_thread(_post)
    embeddings = _normalize_ollama_embed_response(data)

    if embeddings.shape[0] != len(texts):
        raise ValueError(
            f"Embedding count mismatch: got {embeddings.shape[0]} embeddings for {len(texts)} texts"
        )

    return embeddings


async def infer_embedding_dim(embed_model: str, host: str) -> int:
    sample = await custom_ollama_embed(["hello world"], embed_model=embed_model, host=host)
    if sample.size == 0:
        raise ValueError("Failed to infer embedding dimension from Ollama.")
    return int(sample.shape[1])


async def make_lightrag_client(
    working_dir: str,
    ollama_host: str,
    kg_llm_model: str,
    embed_model: str,
):
    if LightRAG is None:
        raise ImportError("LightRAG is not installed or import failed.")

    os.makedirs(working_dir, exist_ok=True)
    embedding_dim = await infer_embedding_dim(embed_model=embed_model, host=ollama_host)

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=ollama_model_complete,
        llm_model_name=kg_llm_model,
        llm_model_max_async=4,
        # llm_model_max_token_size=8192,
        llm_model_kwargs={"host": ollama_host, "options": {"num_ctx": 8192}},
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=lambda texts: custom_ollama_embed(
                texts=texts,
                embed_model=embed_model,
                host=ollama_host,
            ),
        ),
    )

    await rag.initialize_storages()
    return rag


async def ensure_kg_for_problem(
    problem_id: str,
    split: str,
    scienceqa_root: str,
    repo_root: str,
    kg_working_dir: str,
    ollama_host: str,
    kg_llm_model: str,
    embed_model: str,
    threshold: float = 0.20,
    prune_mode: str = "sentence",
    auto_build: bool = True,
):
    image_dir = os.path.join(scienceqa_root, "images", split, str(problem_id))
    if not os.path.isdir(image_dir):
        return

    image_files = find_image_files(image_dir)
    if not image_files:
        return

    texts = []
    for image_path in image_files:
        if auto_build:
            ensure_coe_and_prune_for_image(
                image_path=image_path,
                repo_root=repo_root,
                threshold=threshold,
                mode=prune_mode,
            )
        filtered_txt = final_filtered_txt_path(image_path)
        text = read_text_if_exists(filtered_txt)
        if text:
            texts.append(text)

    if not texts:
        return

    os.makedirs(kg_working_dir, exist_ok=True)
    marker_path = os.path.join(kg_working_dir, ".kg_ready")

    if os.path.exists(marker_path):
        return

    rag = await make_lightrag_client(
        working_dir=kg_working_dir,
        ollama_host=ollama_host,
        kg_llm_model=kg_llm_model,
        embed_model=embed_model,
    )

    combined = "\n\n".join(texts).strip()
    if not combined:
        raise ValueError(f"No filtered text available for KG insertion: pid={problem_id}")

    await rag.ainsert(combined)

    with open(marker_path, "w", encoding="utf-8") as f:
        f.write("ready\n")


async def collect_kg_knowledge_for_problem(
    problem_id: str,
    problem_info: Dict,
    split: str,
    scienceqa_root: str,
    repo_root: str,
    kg_working_dir_base: str,
    ollama_host: str,
    kg_llm_model: str,
    embed_model: str,
    threshold: float = 0.20,
    prune_mode: str = "sentence",
    auto_build: bool = True,
    kg_query_mode: str = "hybrid",
) -> str:
    kg_working_dir = make_problem_kg_working_dir(kg_working_dir_base, split, problem_id)

    await ensure_kg_for_problem(
        problem_id=problem_id,
        split=split,
        scienceqa_root=scienceqa_root,
        repo_root=repo_root,
        kg_working_dir=kg_working_dir,
        ollama_host=ollama_host,
        kg_llm_model=kg_llm_model,
        embed_model=embed_model,
        threshold=threshold,
        prune_mode=prune_mode,
        auto_build=auto_build,
    )

    rag = await make_lightrag_client(
        working_dir=kg_working_dir,
        ollama_host=ollama_host,
        kg_llm_model=kg_llm_model,
        embed_model=embed_model,
    )

    query = problem_info["question"] + "\n" + "\n".join(
        f"{answer_index_to_letter(i)}. {c}" for i, c in enumerate(problem_info["choices"])
    )

    response = await rag.aquery(query, param=QueryParam(mode=kg_query_mode))
    if isinstance(response, str):
        response = response.strip()
    else:
        response = str(response).strip()

    if not response or response in {"[no-context]", "None"}:
        raise ValueError(f"KG query returned empty/no-context result for pid={problem_id}")

    return response


async def evaluate_problem(
    problem_id: str,
    problem_info: Dict,
    split: str,
    scienceqa_root: str,
    repo_root: str,
    ollama_model: str,
    ollama_host: str,
    eval_mode: str,
    auto_build: bool,
    threshold: float,
    prune_mode: str,
    rag_source: str,
    rag_chunk_by: str,
    rag_topk: int,
    kg_working_dir: str,
    kg_llm_model: str,
    kg_query_mode: str,
    kg_embed_model: str,
) -> Dict:
    question = problem_info["question"]
    choices = problem_info["choices"]
    gt_idx = problem_info["answer"]
    gt_letter = answer_index_to_letter(gt_idx)

    knowledge = ""
    retrieval_debug: List[Dict] = []

    if eval_mode == "rag":
        knowledge, retrieval_debug = collect_rag_knowledge_for_problem(
            problem_id=problem_id,
            problem_info=problem_info,
            split=split,
            scienceqa_root=scienceqa_root,
            repo_root=repo_root,
            threshold=threshold,
            prune_mode=prune_mode,
            auto_build=auto_build,
            rag_source=rag_source,
            rag_chunk_by=rag_chunk_by,
            rag_topk=rag_topk,
        )
    elif eval_mode == "kg":
        knowledge = await collect_kg_knowledge_for_problem(
            problem_id=problem_id,
            problem_info=problem_info,
            split=split,
            scienceqa_root=scienceqa_root,
            repo_root=repo_root,
            kg_working_dir_base=kg_working_dir,
            ollama_host=ollama_host,
            kg_llm_model=kg_llm_model,
            embed_model=kg_embed_model,
            threshold=threshold,
            prune_mode=prune_mode,
            auto_build=auto_build,
            kg_query_mode=kg_query_mode,
        )

    prompt = build_question_prompt(
        question=question,
        choices=choices,
        knowledge=knowledge,
        mode=eval_mode,
    )

    raw_output = ask_ollama(
        model=ollama_model,
        prompt=prompt,
        host=ollama_host,
    )
    pred_letter, reason = extract_json_answer(raw_output)
    correct = pred_letter == gt_letter

    result = {
        "problem_id": problem_id,
        "split": split,
        "mode": eval_mode,
        "ground_truth": gt_letter,
        "prediction": pred_letter,
        "correct": correct,
        "question": question,
        "choices": choices,
        "knowledge_chars": len(knowledge),
        "knowledge": knowledge,
        "reason": reason,
        "raw_output": raw_output,
    }
    if retrieval_debug:
        result["retrieval_debug"] = retrieval_debug
    return result


async def amain():
    parser = argparse.ArgumentParser(
        description="Evaluate ScienceQA with baseline, text-only RAG, or KG/LightRAG retrieval."
    )
    parser.add_argument("--repo_root", type=str, default=".")
    parser.add_argument("--scienceqa_root", type=str, default="datasets/ScienceQA/data/scienceqa")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--problem_ids", type=str, default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mode", type=str, choices=["baseline", "rag", "kg"], default="rag")
    parser.add_argument("--auto_build", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--prune_mode", type=str, choices=["word", "sentence", "window"], default="sentence")

    parser.add_argument("--rag_source", type=str, choices=["blip2", "coe", "filtered", "all"], default="all")
    parser.add_argument("--rag_chunk_by", type=str, choices=["sentence", "paragraph"], default="sentence")
    parser.add_argument("--rag_topk", type=int, default=5)

    parser.add_argument("--kg_working_dir", type=str, default="tmp_lightkg")
    parser.add_argument("--kg_llm_model", type=str, default="deepseek-r1:7b")
    parser.add_argument("--kg_embed_model", type=str, default="nomic-embed-text")
    parser.add_argument("--kg_query_mode", type=str, choices=["naive", "local", "global", "hybrid"], default="hybrid")

    parser.add_argument("--ollama_model", type=str, default="deepseek-r1:7b")
    parser.add_argument("--ollama_host", type=str, default="http://localhost:11434")
    parser.add_argument("--output", type=str, default="scienceqa_eval_results.json")

    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    scienceqa_root = os.path.abspath(args.scienceqa_root)

    problems_path = os.path.join(scienceqa_root, "problems.json")
    splits_path = os.path.join(scienceqa_root, "pid_splits.json")

    problems = load_json(problems_path)
    pid_splits = load_json(splits_path)

    if args.problem_ids.strip():
        eval_ids = [x.strip() for x in args.problem_ids.split(",") if x.strip()]
    else:
        eval_ids = pid_splits[args.split]
        if args.limit > 0:
            eval_ids = eval_ids[: args.limit]

    results = []
    num_correct = 0
    num_total = 0

    for pid in eval_ids:
        if pid not in problems:
            print(f"[WARN] problem id not found: {pid}")
            continue

        try:
            item = await evaluate_problem(
                problem_id=pid,
                problem_info=problems[pid],
                split=args.split,
                scienceqa_root=scienceqa_root,
                repo_root=repo_root,
                ollama_model=args.ollama_model,
                ollama_host=args.ollama_host,
                eval_mode=args.mode,
                auto_build=args.auto_build,
                threshold=args.threshold,
                prune_mode=args.prune_mode,
                rag_source=args.rag_source,
                rag_chunk_by=args.rag_chunk_by,
                rag_topk=args.rag_topk,
                kg_working_dir=args.kg_working_dir,
                kg_llm_model=args.kg_llm_model,
                kg_query_mode=args.kg_query_mode,
                kg_embed_model=args.kg_embed_model,
            )
            results.append(item)
            num_total += 1
            if item["correct"]:
                num_correct += 1

            print(
                f"[{num_total}] pid={pid} "
                f"gt={item['ground_truth']} pred={item['prediction']} "
                f"correct={item['correct']} "
                f"knowledge_chars={item['knowledge_chars']}"
            )

        except Exception as e:
            print(f"[ERROR] pid={pid}: {e}")
            results.append(
                {
                    "problem_id": pid,
                    "split": args.split,
                    "mode": args.mode,
                    "error": str(e),
                }
            )

    accuracy = num_correct / num_total if num_total > 0 else 0.0

    summary = {
        "split": args.split,
        "mode": args.mode,
        "num_total": num_total,
        "num_correct": num_correct,
        "accuracy": accuracy,
        "rag_source": args.rag_source if args.mode == "rag" else None,
        "rag_chunk_by": args.rag_chunk_by if args.mode == "rag" else None,
        "rag_topk": args.rag_topk if args.mode == "rag" else None,
        "kg_working_dir": args.kg_working_dir if args.mode == "kg" else None,
        "kg_query_mode": args.kg_query_mode if args.mode == "kg" else None,
        "results": results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== SUMMARY =====")
    print(json.dumps(
        {
            "split": args.split,
            "mode": args.mode,
            "num_total": num_total,
            "num_correct": num_correct,
            "accuracy": accuracy,
            "output": args.output,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    asyncio.run(amain())