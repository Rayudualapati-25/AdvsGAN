# Threat Model

This project treats biomedical text deepfakes as AI-generated or adversarially rewritten abstracts, not factual-verification failures.

In scope:

- machine-generated biomedical abstract text
- paraphrased synthetic text designed to evade a detector
- robustness against style and compression-based rewrites

Out of scope:

- truth verification against underlying papers
- clinical decision support
- video, image, or voice deepfakes

Success condition:

- the robust detector retains stronger attacked-set performance than the screenshot-style baseline

