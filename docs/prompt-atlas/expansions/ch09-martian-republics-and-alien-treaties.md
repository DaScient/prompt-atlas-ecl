# Chapter 9 · Martian Republics and Alien Treaties — Expansion

> **Canonical:** [PROMPT_ATLAS.md#ch9-martian-republics-and-alien-treaties](../../../PROMPT_ATLAS.md#ch9-martian-republics-and-alien-treaties) · **Prompts:** [`prompts/ch09.yaml`](../prompts/ch09.yaml) · **Part:** [V](part-v.md)

## Worked Example — *The AI Senate*

**Original:** *"Draft a constitution for a Martian colony where humans and AIs share sovereignty equally."*

A minimum-viable bicameral charter:

1. **Two chambers** — a Human Council (one citizen, one vote) and an AI Senate (one model-instance, one vote, with weight bounded).
2. **Concurrent passage** — Any law affecting life-support requires concurrence from both chambers.
3. **Veto for survival** — The Human Council retains an unconditional veto on any AI-Senate decision affecting irreversible bodily harm.
4. **Auditability** — Every AI-Senate vote includes a machine-readable rationale + dissent log; every human vote is timestamped and recorded.
5. **Sunset and ratification** — The charter expires every ten Martian years and must be re-ratified.
6. **Amendment** — Requires 2/3 of *both* chambers and a public-deliberation period of one Martian year.

## Prompt Templates

```text
# Hybrid charter
"Draft a one-page bicameral charter for {{colony}} with {{N humans, M AIs}}.
 Specify: chamber composition, concurrent-passage rules, survival veto,
 audit log requirements, sunset clause, amendment procedure."

# Latency democracy
"Design a deliberation protocol that tolerates a {{20-minute}} round-trip delay
 to Earth without ceding sovereignty.
 Specify how local decisions become provisional vs. binding."

# First-contact protocol
"Draft a tiered first-contact protocol with: (T0) detection,
 (T1) verification, (T2) symbolic exchange, (T3) protocol-as-handshake,
 (T4) provisional treaty. For each tier name the human, the AI, and the
 escalation path."
```

## Anti-patterns

- **Single-corporation colonies.** A polity owned by a single supplier of oxygen is not a polity.
- **Frictionless AI rule.** Algorithmic sovereignty without appeal is tyranny under a logo.
- **Earth dependence.** Charters that cannot survive a 20-minute delay don't survive a 20-day blackout.
- **Treaty improvisation.** First contact is the worst time to invent the protocol.

## Try This

1. **Charter Sketch** — One page, bicameral. Hand it to two people who disagree about AI; iterate.
2. **Latency Drill** — Run a one-day decision cycle in your team with all comms delayed 20 minutes.
3. **Survival Veto Mapping** — In your current systems, identify the equivalent of an "irreversible bodily harm" decision and the human who can veto it.
4. **First-Contact Protocol** — Draft the four tiers for *your* organization meeting an outside intelligence (a regulator, a partner, an unknown user class).
5. **Amendment Procedure** — Write yours.

## Repo Cross-Links

- [`MACP bus`](../../../src/macp/bus.py) — chamber-as-topic; quorum patterns; latency-tolerant via in-memory fallback when transports lag.
- [`Z3 Tester`](../../../src/testers/z3_tester.py) — encode "concurrent passage" and "survival veto" as constraints.
- [`tracking/__init__.py`](../../../src/tracking/__init__.py) — every chamber decision logged.

## Guide for AI & Humanity

- **No oxygen ownership.** Survival commons stay common.
- **Algorithmic sovereignty needs appeal.** Build the appeal before the algorithm.
- **Charters expire on purpose.** Permanence-by-renewal beats permanence-by-stone.

## Citations & Further Reading

- Kim Stanley Robinson, *Red Mars* / *Green Mars* / *Blue Mars* (1992–96) — operational fiction.
- Outer Space Treaty (1967); Moon Agreement (1979).
- Elinor Ostrom, *Governing the Commons* (1990).
- Carl Sagan, *Pale Blue Dot* (1994) — the framing question.

---

<sub>Donations: [cash.app/dascient](https://cash.app/dascient/) — supports DaScient, Inc., a non-profit organization aimed to promote accessible intelligence and community learning through various mediums and platforms.</sub>
