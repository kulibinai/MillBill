# MillBill collection

This directory is the collection of milling stocks described in Sec. 4.1 of MillBill. It contains 4,294 single-solid parts drawn from the GrabCAD Community Library (https://grabcad.com/library). Every part is paired with one milling stock. The part STEP files are not in this directory.

A MillBill task, as stated in the paper, is the part as a STEP boundary representation, the stock paired with it here, and the workpiece material. The default tolerance used in the paper, 0.1 mm, is a parameter of the evaluation. It is not stored in these files.

## Stocks

Of the 4,294 stocks, 1,396 are turned and 2,898 are prismatic. The counts below are the counts in this directory.

Turned stocks, under `turned stock/<id>/`:

| Kind | Parts | Construction |
|---|---:|---|
| Stepped shaft | 693 | 1 mm of allowance |
| Solid of revolution | 703 | No allowance; local cavities filled out to the circumscribed circle |

Prismatic stocks, under `prismatic stock/<kind>/<id>/`. A stepped casting, forging, or profile, or a rolled flat bar, square bar, plate, plate strip, or sheet:

| Directory | Kind in the paper | Parts |
|---|---|---:|
| `casting` | stepped casting | 593 |
| `forging` | forging | 409 |
| `profile` | profile | 158 |
| `flat bar` | rolled flat bar | 658 |
| `square` | square bar | 203 |
| `plate` | plate | 457 |
| `plate strip` | plate strip | 134 |
| `sheet` | sheet | 286 |

The 100-part benchmark in the paper is a stratified draw from this collection, with random seed 2026: ten parts with a turned stock and ten from each of the eight prismatic kinds, then ten more from the remaining candidates. [`evaluation.json`](evaluation.json) lists those 100 parts. Each entry names the `blank.step` and `part.json` of that part, with paths relative to this directory. The metrics in the paper use a tolerance of 0.1 mm, which is recorded in that file. The draw is not copied out as a separate directory.

## Files

Each part directory contains only:

| File | Contents |
|---|---|
| `blank.step` | The milling stock, STEP, in millimetres |
| `part.json` | The GrabCAD page of the source model, and the workpiece material |

```json
{
  "grabcad_url": "https://grabcad.com/library/...",
  "material": "Ti-6Al-4V"
}
```

The material is the workpiece material specified by the author of the model on GrabCAD. The URL is the library page of the source model. For 21 parts that page was not stored, and the URL is formed from the GrabCAD slug; those links were not verified against the live library.

## GrabCAD

The parts were taken from the GrabCAD Community Library under the GrabCAD Website Terms of Use (https://grabcad.com/terms). The original STEP models are not included. Each `part.json` gives the page of the source model and names its author by that link. The stocks are derived from those models for this study. Rights in the source models remain with their authors and with GrabCAD; nothing here licenses them.
