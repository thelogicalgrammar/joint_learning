"""
Helpers for the paper-facing scripts (analyses/hmm/03_condition_effects.py,
05_ppc.py): numbers in the precision used in the paper, and a report that
echoes each line to stdout and collects it into a Markdown file.

    num, summ          — one number / 'median [2.5%, 97.5%]' of posterior draws
    fmt_p              — a probability, printed as '> 0.999' / '< 0.001' at the ends
    p_tex, p_rel       — fmt_p strings as a LaTeX table cell / right-hand side of 'P(...)'
    Report             — collect lines (Markdown, LaTeX blocks) and write them to a file
"""
import numpy as np


def num(x, kind):
    """One number in the precision used in the paper."""
    if kind == 'gamma':                              # 3 significant digits
        return f'{x:.{max(0, 2 - int(np.floor(np.log10(abs(x)))))}f}'
    if kind == 'trials':
        return f'{x:.0f}' if x >= 100 else f'{x:.1f}'
    if kind == 'kappa':
        return f'{x:.2f}'
    if kind in ('lambda', 'rate', 'index'):
        return f'{x:.3f}'
    if kind == 'diff':                               # signed rate difference
        return f'{x:+.3f}'
    if kind == 'ratio':
        return f'{x:.0f}' if x >= 100 else f'{x:.1f}' if x >= 10 else f'{x:.2f}'
    if kind == 'loglik':
        return f'{x:.0f}'
    raise ValueError(kind)


def summ(x, kind):
    """'median [2.5%, 97.5%]' of posterior draws x."""
    lo, md, hi = np.percentile(x, [2.5, 50, 97.5])
    return f'{num(md, kind)} [{num(lo, kind)}, {num(hi, kind)}]'


def fmt_p(p):
    """A probability estimated from posterior draws."""
    if p > 0.999:
        return '> 0.999'
    if p < 0.001:
        return '< 0.001'
    return f'{p:.3f}'


def p_tex(s):
    """fmt_p string as a LaTeX table cell."""
    return f'${s}$' if s[0] in '<>' else s


def p_rel(s):
    """fmt_p string as the right-hand side of 'P(...)': '= 0.930' or '> 0.999'."""
    return s if s[0] in '<>' else f'= {s}'


class Report:
    """Lines of a Markdown report; each line is also printed."""

    def __init__(self):
        self.lines = []

    def out(self, line=''):
        print(line)
        self.lines.append(line)

    def latex(self, lines):
        """A fenced LaTeX block, ready to paste into the paper."""
        self.out('```latex')
        for line in lines:
            self.out(line)
        self.out('```')
        self.out()

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(self.lines) + '\n')
        print(f'\nSaved: {path}')
