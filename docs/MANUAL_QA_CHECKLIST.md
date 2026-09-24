# Final Manual Question QA

Run this only after the cleanup validator passes and DocuBot starts normally from VS Code.

Use real questions, not only certification prompts. Cover at least:

1. Direct MISRA rule question.
2. MISRA yes/no compliance question with code context.
3. Explanation/why question that should use grounded Qwen generation.
4. Multi-part or comparison question.
5. Natural English paraphrase without exact rule wording.
6. Follow-up question that depends on the previous turn.
7. Ambiguous question where DocuBot should ask/answer conservatively.
8. Unsupported/fake rule reference that should fall back safely.
9. Out-of-scope/current-world question that should not hallucinate.
10. Repeat a previously slow complex question and record the observed response time.

For each answer, check: semantic correctness, grounding, source relevance, natural wording, fallback behavior, and obvious latency regression.
