---
title: oncomap
emoji: 🧬
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# oncomap API

Compound (SMILES) -> protein targets -> cancer cell lines -> cancer types.
Interactive docs at `/docs`. All endpoints except `/health` require the
`X-API-Key` header when the `ONCOMAP_API_KEY` Space secret is set.
