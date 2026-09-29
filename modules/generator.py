# Generator: Retrieved context ko prompt me daal kar factual answer banwana (No Hallucination)
from typing import List, Dict, Optional
import openai


# Grounded prompt assemble karna (Sources citations ke sath)
def build_grounded_prompt(user_question: str, retrieved_chunks: List[str], custom_system: Optional[str] = None) -> List[Dict[str, str]]:
    context = "\n\n".join([f"[Source {i+1}]\n{c.strip()}" for i, c in enumerate(retrieved_chunks)])
    sys_prompt = custom_system or (
        "Answer ONLY using the provided [CONTEXT]. Cite source numbers (e.g. [Source 1]). "
        "If not present in context, state: 'The provided document context does not contain this information.'"
    )
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"[CONTEXT]\n{context}\n[/CONTEXT]\n\nQuestion:\n{user_question}"}
    ]


# LLM ko call karke answer generate karwana
def generate_answer(messages: List[Dict[str, str]], client: openai.OpenAI, model: str = "openai/gpt-4o-mini", temp: float = 0.2) -> str:
    res = client.chat.completions.create(model=model, messages=messages, temperature=temp)
    return res.choices[0].message.content
