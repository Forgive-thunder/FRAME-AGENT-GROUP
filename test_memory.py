"""Quick test for the FAISS long-term memory + Filter topic-feasibility check.

Run from the project root::

    python test_memory.py

Flow:
  1. Build/update the FAISS index from ``output_dataset/stage2_results.json``.
  2. Query the memory with a hypothetical medical topic.
  3. Run the Filter agent to judge whether the topic has historical support.

The Filter's final judgment uses a fake LLM by default (no API key needed);
swap in ``llm=LLMClient()`` to use the real model.  The FAISS retrieval itself
uses a real sentence-transformers embedder (``config.EMBEDDING_MODEL``).
"""
from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from memory.faiss_manager import MemoryManager
from agents.filter import FilterAgent
from llm.client import LLMClient  # noqa: F401 — import to enable the real-model swap


TOPIC = "抗生素耐药性在城市医疗系统中的传播机制与干预策略"


class FakeLLM:
    """Deterministic stand-in for the Filter's final judgment (no API needed)."""

    def chat_json(self, system, user, temperature=None):
        return {
            "supported": True,
            "reason": "检索到足够的历史 Reflector 训练数据支撑该课题。",
            "outline": (
                "一、研究背景：抗生素耐药性的全球威胁\n"
                "二、研究目标：刻画城市医疗系统中的耐药性传播路径\n"
                "三、数据与方法：结合历史反思数据与问答对\n"
                "四、预期贡献：提出可落地的干预策略"
            ),
        }


def main() -> None:
    memory = MemoryManager()

    print("== 1. 构建/更新 FAISS 记忆 ==")
    n = memory.build_or_update_index()
    print(f"   已索引 {n} 条 chunk\n")

    print(f"== 2. 检索课题：{TOPIC} ==")
    retrieved = memory.query_memory(TOPIC, top_k=3)
    if not retrieved:
        print("   （无检索结果，索引可能为空）")
    for r in retrieved:
        print(f"   score={float(r.get('score', 0.0)):.3f}  source={r.get('source')}")
        print(f"      {str(r.get('text', ''))[:120]}...")
    print()

    print("== 3. Filter Agent 最终判断 ==")
    # 使用假 LLM 演示流程；接入真实模型时改为 llm=LLMClient()
    filter_agent = FilterAgent(llm=FakeLLM(), memory=memory, top_k=3)
    result = filter_agent.run({"topic": TOPIC})
    print(f"   verdict   : {result['verdict']}")
    print(f"   supported : {result['supported']}")
    print(f"   reason    : {result['reason']}")
    print(f"   outline   :\n{result['outline']}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"\n[错误] {exc}")
        print("请先安装依赖：pip install faiss-cpu numpy sentence-transformers")
        sys.exit(1)
