# Benchmark Report

## Summary

Recorded 2 run(s). Fastest run: `multi-agent` (58.81s). Best heuristic quality: `multi-agent` (10.0/10).

## Metrics

| Run | Latency (s) | Cost (USD) | Quality | Notes |
|---|---:|---:|---:|---|
| baseline | 148.32 | 0.0005 | 4.0 | status=ok; routes=none; sources=0; citation_coverage=0%; errors=0 |
| multi-agent | 58.81 | 0.0022 | 10.0 | status=ok; routes=researcher>analyst>writer>critic>done; sources=5; citation_coverage=100%; errors=0; trace=https://smith.langchain.com/o/c47b752a-42ee-4fe2-af72-7b4d227b2291/projects/p/d7934dad-15f1-400c-9f34-1cab38a8ac5b/r/8560b08e-249f-4b2f-9274-9e0901af378d?poll=true |

## Notes

- `baseline`: status=ok; routes=none; sources=0; citation_coverage=0%; errors=0
- `multi-agent`: status=ok; routes=researcher>analyst>writer>critic>done; sources=5; citation_coverage=100%; errors=0; trace=https://smith.langchain.com/o/c47b752a-42ee-4fe2-af72-7b4d227b2291/projects/p/d7934dad-15f1-400c-9f34-1cab38a8ac5b/r/8560b08e-249f-4b2f-9274-9e0901af378d?poll=true
