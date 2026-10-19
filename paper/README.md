# Paper draft

`main.tex` is an ICLR-style draft of *When Unusual Is Not Unsafe: Intervention-Response Monitoring
under Benign Distribution Shift*.

**Evidence status.** No language-model experiment has been run. Every red `\TBD{...}` cell or sentence
is a placeholder for a value produced by `irmon evaluate` / `irmon ablate` / `irmon control` /
`irmon attack`. The only numbers reported come from the restricted-model simulation
(`scripts/run_theory_sim.py`, Section 4.6 and Figure 2) and the generator/edit validity checks
(Appendix B).

Compile on Overleaf or locally: upload `main.tex`, `references.bib` and `figures/`, then run
`pdflatex main && bibtex main && pdflatex main && pdflatex main`. For submission, swap the preamble
for the official `iclr2027_conference.sty`.
