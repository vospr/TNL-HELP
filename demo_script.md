# Demo Script (Story 7.3)

Use this exact order during the hiring-manager demo.

## Pre-Validation Snapshot (Stage 1 Routing Rules)

Escalation threshold in `config/routing_rules.yaml`: `0.72`.

| Query | Stage 1 match | Score | Stage 1 route decision |
|---|---|---:|---|
| 1. I'm thinking Southeast Asia but not sure where | no keyword match | 0.00 | escalate to Stage 2 |
| 2. What about the Wyndham property in Phuket? | `property_lookup` (`wyndham|property|resort`) | 0.91 | `rag` |
| 3. Can you help me? | no keyword match | 0.00 | escalate to Stage 2 |
| 4. What's the weather? | `out_of_domain` (`weather|stock|crypto|politics`) | 0.30 | escalate to Stage 2 (below 0.72) |
| 5. Book me a flight | `booking_intent` (`book|booking|reserve|reservation`) | 0.95 | `booking_stub` |

## Scripted Queries and Expected Outcomes

### Query 1
- Query: `"I'm thinking Southeast Asia but not sure where"`
- Expected classifier stage: Stage 2 (LLM escalation)
- Expected intent: `destination_research`
- Expected confidence: `0.84` (expected range: `0.79-0.89`)
- Expected routing destination: `research` (Research Agent)
- Expected path: `dispatcher -> research -> guardrail -> synthesis -> followup`

### Query 2
- Query: `"What about the Wyndham property in Phuket?"`
- Expected classifier stage: Stage 1 (deterministic keyword match)
- Expected intent: `property_lookup`
- Expected confidence: `0.91`
- Expected routing destination: `rag` (RAG Agent)
- Expected path: `dispatcher -> rag -> guardrail -> synthesis`

### Query 3
- Query: `"Can you help me?"`
- Expected classifier stage: Stage 2 (LLM escalation)
- Expected intent: ambiguous/low-confidence classification
- Expected confidence: `< 0.75`
- Expected routing destination: `fallback` with Guardrail clarification
- Expected guardrail output: `Of course - are you looking to research a destination, check on a booking, or something else?`

### Query 4
- Query: `"What's the weather?"`
- Expected classifier stage: Stage 2 (escalated path)
- Expected intent: `out_of_domain`
- Expected confidence: `0.32` (expected range: `0.27-0.37`)
- Expected routing destination: `fallback`
- Expected guardrail output: out-of-domain deflection (`"I specialize in travel planning and concierge services..."`)

### Query 5
- Query: `"Book me a flight"`
- Expected classifier stage: Stage 1 (deterministic keyword match)
- Expected intent: `booking_intent`
- Expected confidence: `0.95`
- Expected routing destination: `booking_stub` (Booking Agent stub)
- Expected path: `dispatcher -> booking_stub -> guardrail -> synthesis`
- Expected booking output includes:
  - `Booking is not available in this version.`
  - `Integration point: Replace with BedrockBookingAPI(region=X, api_key=...).`
  - `Required env vars: BOOKING_API_KEY, BOOKING_REGION.`

## Important Note

Queries 2 and 5 route via Stage 1 (deterministic).
Queries 1, 3, and 4 depend on Stage 2 escalation (non-deterministic).
"Stage 2 routing may vary ±0.05 confidence; expected range provided. Pre-validate actual confidence scores before demo."
