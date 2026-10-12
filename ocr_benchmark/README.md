# Frame 137 OCR comparison

Recording: `recordings/20261005_195417.db3`. RGB rotated clockwise as supplied
by the reader. Five object rectangles from tabletop segmentation, including
the 5-depth-pixel bottom extension. Saved PNGs are the original-resolution crops.
Both engines ran locally on CPU, with four inference threads, 2x cubic
enlargement, four crop orientations, and a minimum reported confidence of 0.3.

The table reports meaningful actual readings across orientations, not
manually corrected OCR outputs. `results.json` contains all readings, scores,
polygons, crop bounds, and individual inference timings (including noise).

| Crop, left to right | Visually identifiable main label | RapidOCR 1.4.4 | EasyOCR 1.7.2 (English/Spanish) |
|---|---|---|---|
| 0 | Melatonina | `Melatonina` | `Molotonlua` |
| 1 | IBUPROFENO | `IBUPROFENO` | `IRUPROFENO` |
| 2 | Colgate | `Colgate'` (extra apostrophe) | `Colgate` |
| 3 | NESCAFÉ jar | `NESOARE` (incorrect) | No reliable label text |
| 4 | Purple cup, no visible printed label | No text | No text |

Tiny supplementary text was unreliable in both engines. RapidOCR's bundled
Chinese/English recognition model also emitted a spurious Chinese character
on the jar. Large confidence scores on isolated characters were not evidence
of useful label recognition. Neither engine reliably extracted dosage text.

## Timing on this laptop

| Engine | Total inference, 5 crops × 4 orientations | Mean per crop/orientation | Initialization |
|---|---|---|---|
| RapidOCR | 10.76 seconds | 538 ms | 1.72 seconds |
| EasyOCR | 11.93 seconds | 596 ms | 14.10 seconds, including first-run model downloads |

Inference excludes loading the bag, segmentation, projection, crop preparation,
and model initialization. These are single benchmark runs, not a statistical
performance study. Keep the initialized engine alive for repeated requests.

RapidOCR is the better baseline in this experiment, but exact product identity
and dosage should not be accepted from one OCR reading. Rotating crops helps
vertical labels; enlarged pixels do not restore lost detail. A closer, sharper
view or multi-frame confirmation is needed for reliable small-print reading.
