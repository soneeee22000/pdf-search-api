# Test fixtures

Two tiny generated PDFs, committed so the test suite needs no network access and no external corpus.
They are synthetic — no client documents are in this repository.

| File                  | What it exercises                                                                                                                                                                                           |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `deliberation_fr.pdf` | Two pages of French administrative prose with a repeated header and a `Page N sur M` footer. Covers page numbering, per-page attribution, CRLF handling, and the survival of an `L. 2121-29` legal citation |
| `no_text_layer.pdf`   | One page containing only a filled rectangle. The scanned-document case: extraction must report `no_text` rather than crashing or silently indexing nothing                                                  |

Regenerate with `reportlab` (a dev-time tool only, not a runtime dependency):

```python
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

PAGES = [
    ("Commune de Saint-Martin", [
        "EXTRAIT DU REGISTRE DES DELIBERATIONS",
        "Vu l'article L. 2121-29 du code general des collectivites",
        "territoriales, le conseil municipal regle par ses deliberations",
        "les affaires de la commune.",
        "Le conseil municipal approuve la convention de mecenat",
        "financier et autorise Monsieur le Maire a la signer.",
    ]),
    ("Commune de Saint-Martin", [
        "ORDRE DU JOUR DE LA SEANCE DU 18 JUIN 2026",
        "1. Approbation du proces-verbal de la seance precedente",
        "2. Declaration prealable DP-26A0019, place des Hauts Taillis",
        "3. Budget primitif 2026 - decision modificative n 2",
        "4. Questions diverses",
    ]),
]

c = canvas.Canvas("deliberation_fr.pdf", pagesize=A4)
width, height = A4
for index, (header, lines) in enumerate(PAGES, start=1):
    c.setFont("Helvetica", 9)
    c.drawString(60, height - 50, header)
    c.setFont("Helvetica", 11)
    y = height - 100
    for line in lines:
        c.drawString(60, y, line)
        y -= 20
    c.setFont("Helvetica", 8)
    c.drawString(60, 40, f"Page {index} sur {len(PAGES)}")
    c.showPage()
c.save()

c = canvas.Canvas("no_text_layer.pdf", pagesize=A4)
c.rect(80, 400, 400, 200, fill=1)
c.showPage()
c.save()
```
