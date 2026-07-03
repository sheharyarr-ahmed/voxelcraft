# License compliance

Every model artifact this app touches — base weights, LoRAs, ControlNet, annotators — gets a
verified license record before it is referenced in code. This is the protocol. It exists because
the repo is public and the README claims every license is documented; one unverified entry breaks
that claim.

## Registry contract (`src/config.py`)

Every `LORA_REGISTRY` entry carries all four fields. No partial entries, ever:

- `url` — the model card URL (civitai or HF Hub), not a mirror, not a download link.
- `author` — the creator name exactly as displayed on the model card.
- `license` — the exact license name shown on the card (e.g. `CreativeML Open RAIL-M`), or for
  civitai the permission set, since civitai models often have no named license.
- `commercial_use` — `True` only after manual verification by Sheharyar. Defaults to `False`.

## Who verifies (D4 — hard pause point)

Verification is performed by Sheharyar personally. Tooling and agents — including Claude Code —
never set `commercial_use=True`, never infer it from a license name, and never mark verification
done. When a new LoRA is proposed: stop, hand the model card URL to Sheharyar, and wait. This is
a vetting-call question and the answer has to be his own.

## civitai protocol

1. Open the model card at the recorded `url`.
2. Read the license/permissions block — the commercial-use flags civitai renders per model
   ("Sell images they generate", "Use on generation services", "Sell this model or merges").
3. Quote the exact permission wording in a comment next to the registry entry. Paraphrases drift;
   quotes don't.
4. Record the model card URL and the date checked in the same comment. civitai permissions can
   change after upload; the date bounds what was verified.

## HF Hub protocol

Two sources, both recorded: the `license` field in the model card YAML metadata, and any
`LICENSE` / `LICENSE.md` file committed in the weights repo itself. If the metadata tag and the
repo file disagree, that is a contradiction — apply the ambiguity rule below.

## Base artifacts (documented, not just LoRAs)

The license table covers everything loaded at runtime, not only registry LoRAs:

- **SD 1.5 base weights** (`stable-diffusion-v1-5/stable-diffusion-v1-5`) — CreativeML Open
  RAIL-M. Note its use-based restriction clauses (Attachment A): commercial use is permitted, but
  the enumerated prohibited uses bind downstream users and must be passed along, which the README
  license table acknowledges.
- **ControlNet OpenPose weights** (`lllyasviel/control_v11p_sd15_openpose`) — record the
  card's license tag and the repo license file per the HF Hub protocol.
- **controlnet-aux OpenPose annotator weights** — the `controlnet-aux` pip package license covers
  code only. The pose-detection checkpoints it downloads (from `lllyasviel/Annotators`) are
  separate artifacts with their own provenance; record the license stated on that repo. If the
  repo states none, that is ambiguity — escalate to Sheharyar, do not assume.

## Enforcement

- `tests/test_config.py` asserts every registry entry carries all four fields and that
  `commercial_use is True` — a `False` or missing value fails the suite, which runs on the Stop
  hook via `.claude/verify.sh`.
- The README carries the full license table in Phase 4: artifact, model card URL, author, license,
  date verified. The table is generated from the registry plus the base artifacts above, so the
  two can't diverge silently.

## Rule of ambiguity

An unclear, contradictory, or research-only license means the model is not used. Not "used with a
disclaimer", not "used until someone objects" — not used. There are enough permissively licensed
LoRAs that no single model is worth the exposure. No exceptions.
