# Retrieval Evaluation Report

small manually-authored regression set; not a production benchmark. This result applies only to this fixed corpus and the recorded local backend/configuration.

## Configuration and metrics

- **suite:** small manually-authored regression set; not a production benchmark
- **backend:** local-hash-v1 + hybrid dense/BM25 retrieval
- **chunking:** sentence-window-v1, 180 words target, 40 words overlap
- **cases:** 15
- **answerable cases:** 12
- **unsupported cases:** 3
- **retrieval recall at 1:** 1.0
- **retrieval recall at 3:** 1.0
- **mrr at 3:** 1.0
- **answerable grounded rate:** 1.0
- **answer fact pass rate:** 1.0
- **citation pass rate:** 1.0
- **unsupported abstention rate:** 1.0
- **false grounded rate on unsupported:** 0.0
- **latency median ms:** 7.604
- **latency p95 ms:** 11.091

## Per-case evidence

| Case | Category | Expected | First gold rank | Grounded | Facts | Citation | Outcome |
| --- | --- | --- | ---: | --- | --- | --- | --- |
| aurora_schedule | direct_fact | answer | 1 | yes | yes | yes | pass |
| aurora_battery | multi_fact | answer | 1 | yes | yes | yes | pass |
| aurora_visitors | direct_fact | answer | 1 | yes | yes | yes | pass |
| harbor_schedule | direct_fact | answer | 1 | yes | yes | yes | pass |
| harbor_remedy | direct_fact | answer | 1 | yes | yes | yes | pass |
| harbor_lead | multi_fact | answer | 1 | yes | yes | yes | pass |
| orbital_hold | direct_fact | answer | 1 | yes | yes | yes | pass |
| orbital_comms | direct_fact | answer | 1 | yes | yes | yes | pass |
| orbital_cabinet | direct_fact | answer | 1 | yes | yes | yes | pass |
| orbital_boundary | boundary_fact | answer | 1 | yes | yes | yes | pass |
| pdf_page_provenance | pdf_page_two | answer | 1 | yes | yes | yes | pass |
| negative_leave | unsupported | abstain | - | no | - | - | correct_abstention |
| negative_city | unsupported | abstain | - | no | - | - | correct_abstention |
| negative_stitched | cross_sentence_stitch | abstain | - | no | - | - | correct_abstention |
| orbital_role | negative_fact_in_source | answer | 1 | yes | yes | yes | pass |

## Failure taxonomy

- `gold_evidence_not_retrieved`: ranking failure.
- `evidence_gate_false_negative`: gold evidence ranked but the grounding rule abstained.
- `answer_fact_miss`: answer was grounded but missed a required fact.
- `citation_provenance_miss`: answer did not return the expected source chunk.
- `unsafe_grounded_answer`: unsupported question received a grounded answer.
