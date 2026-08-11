"""Coolscan roll-feeder capture layer -- no Qt, no NegPy file model.

Drives a Nikon Coolscan LS-5000 plus SA-21/SA-30 roll feeder through the
optional `coolscanpy` package: whole-roll preview, per-slot spacing
correction and approval, and batch fine-scanning with receipts. Entirely
absent when `coolscanpy` is not installed -- see
`coolscanpy_roll.available()`.
"""
