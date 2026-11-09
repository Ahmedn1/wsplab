# Logo

A 5×5 grid of weights. The lit ones draw a W, and the centre one, in blue, is the edited weight.

| File | Use |
|---|---|
| `logo-light.svg`, `logo-dark.svg` | Mark and wordmark, for light and dark backgrounds |
| `mark-light.svg`, `mark-dark.svg` | The mark alone, for icons and small spaces |
| `avatar.png` | 512×512, for a GitHub or Hugging Face avatar |
| `social-preview.png` | 1280×640, GitHub's social preview |

- **Colours:** ink `#16181D` on light and `#F3F3EF` on dark; blue `#2563EB` on light and `#3B82F6` on dark. Weights
  that are off show at 12% of the ink colour.
- **Wordmark:** [DM Mono](https://github.com/googlefonts/dm-mono) Medium ("wsp") and Light ("lab", 60%), tracked
  −0.05 em, converted to outlines. DM Mono is licensed under the SIL Open Font License 1.1.
- **Rebuilding:** `make_logo.py` writes every file here. Its docstring says where to get the fonts.
