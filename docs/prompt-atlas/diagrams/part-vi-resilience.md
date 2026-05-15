# Part VI — Resilience and Survival · Diagram

```mermaid
flowchart TB
  Shock([Shock]) --> Sentinel[[AI Sentinel]]
  Sentinel -->|warn| Decision{Listen?}
  Decision -->|yes| Absorb[Absorb / Reroute]
  Decision -->|no| Collapse((Collapse))
  Absorb --> Renewal[(Renewal)]
  Collapse --> Renewal
  Renewal --> Memory[/Atlas of Renewal/]
  Memory --> Permanence{{Designed Permanence}}
  Permanence -->|self-healing| Archive[(Self-healing Archive)]
  Permanence -->|adaptive| Constitution[Living Constitution]
  Archive -. anchors .- Memory
  Constitution -. anchors .- Memory
```

*Shocks are met by the AI sentinel; whether or not foresight is heeded, the path leads through renewal into designed permanence (archives + adaptive constitutions).* Anchored to [Chapter 11](../../../PROMPT_ATLAS.md#ch11-preparing-for-collapse-and-renewal) and [Chapter 12](../../../PROMPT_ATLAS.md#ch12-designing-permanence).
