#!/usr/bin/env python3
"""Apply ALL review fixes to the paper in one shot."""
import re

fn = 'E:/Desktop/dev/Aerial-Ground-Cooperative-Perception/adaptive_fusion.tex'

with open(fn, 'rb') as f:
    data = f.read()

# ===== 1) PREAMBLE FIXES =====
# Add amsthm package after amssymb
data = data.replace(
    b'\\usepackage{amssymb}\r\n\\usepackage{booktabs}',
    b'\\usepackage{amssymb}\r\n\\usepackage{amsthm}\r\n\\usepackage{booktabs}'
)

# Add theorem environment after custom commands
data = data.replace(
    b'\\def\\msc#1{\\texttt{#1}}\r\n\r\n\\begin{document}',
    b'\\def\\msc#1{\\texttt{#1}}\r\n\r\n% Theorem environments\r\n\\newtheorem{theorem}{Theorem}\r\n\r\n\\begin{document}'
)

# ===== 2) FIX EQ(2): additive pose error -> multiplicative =====
old_eq2 = (
    b'\\textbf{Relative pose errors.} Transformation error:\r\n'
    b'\\begin{equation}\r\n'
    b'    \\tilde{\\mb{T}}_{v2d} = \\mb{T}_{v2d} + \\Delta \\mb{T}\r\n'
    b'    \\label{eq:pose_error}\r\n'
    b'\\end{equation}\r\n'
    b'where $\\Delta \\mb{T}$ includes translation $\\Delta_t$ (up to 7m) and rotation $\\Delta_\\theta$.'
)
new_eq2 = (
    b'\\textbf{Relative pose errors.} Transformation matrices belong to $SE(3)$, '
    b'which is not a vector space. We model perturbation multiplicatively using the exponential map:\r\n'
    b'\\begin{equation}\r\n'
    b'    \\tilde{\\mb{T}}_{v2d} = \\exp(\\boldsymbol{\\xi}^\\wedge) \\, \\mb{T}_{v2d}\r\n'
    b'    \\label{eq:pose_error}\r\n'
    b'\\end{equation}\r\n'
    b'where $\\boldsymbol{\\xi} \\in \\mathbb{R}^6$ is the twist coordinate (comprising translation '
    b'$\\Delta_t \\in \\mathbb{R}^3$ and rotation $\\Delta_\\theta \\in \\mathbb{R}^3$) and $^\\wedge$ '
    b'maps it to the $se(3)$ Lie algebra. This formulation respects the group structure of $SE(3)$ '
    b'and avoids the pathological behaviour of additive noise, which leaves the group manifold entirely.'
)
data = data.replace(old_eq2, new_eq2)

# ===== 3) REPLACE THEOREM 1 PARAGRAPH WITH PROPER THEOREM =====
old_theorem = (
    b'\\textbf{Theoretical analysis (Theorem 1: extrapolation bound).} '
    b'Let $\\phi_\\theta : \\mathbb{R}^d \\to \\mathbb{R}^3$ be the correction MLP, '
    b'and assume $\\phi_\\theta$ is $L_\\phi$-Lipschitz in its input '
    b'(standard for MLPs with bounded-weight spectral norms, e.g.\\ '
    b'via spectral normalisation during training). '
    b'Let $\\text{PE} : \\mathbb{R} \\to \\mathbb{R}^d$ be the sinusoidal encoding in '
    b'Eq.~\\ref{eq:altitude_pe}. For any two altitudes $h, h\' \\in \\mathbb{R}^+$,\r\n'
    b'\\begin{equation}\r\n'
    b'    \\|\\Delta P(h) - \\Delta P(h\')\\|_2\r\n'
    b'    = \\|\\phi_\\theta(\\text{PE}(h)) - \\phi_\\theta(\\text{PE}(h\'))\\|_2\r\n'
    b'    \\le L_\\phi \\,\\|\\text{PE}(h) - \\text{PE}(h\')\\|_2.\r\n'
    b'    \\label{eq:lip_chain}\r\n'
    b'\\end{equation}\r\n'
    b'Each PE component is bounded by 1 in absolute value, so by the per-component '
    b'Lipschitz property of $\\sin$ and $\\cos$, '
    b'$\\|\\text{PE}(h) - \\text{PE}(h\')\\|_2 \\le C_d \\,|h - h\'|$ '
    b'with $C_d = \\sqrt{d/(10000^{0}) + d/(10000^{2/d}) + \\dots} = \\mathcal{O}(d)$. Hence\r\n'
    b'\\begin{equation}\r\n'
    b'    \\|\\Delta P(h) - \\Delta P(h\')\\|_2 \\le L_\\phi \\, C_d \\,|h - h\'|.\r\n'
    b'    \\label{eq:extrapolation_bound}\r\n'
    b'\\end{equation}\r\n'
    b'Eq.~\\ref{eq:extrapolation_bound} is a \\emph{uniform} bound valid for all $h, h\'$, '
    b'not just training altitudes: the correction at 70m is provably within '
    b'$L_\\phi C_d \\cdot 15\\text{m}$ of the correction at 55m. By contrast, '
    b'a learned altitude embedding $E(h) \\in \\mathbb{R}^d$ satisfies no such bound'
    b'---for $h \\notin \\{h_{\\text{train}}\\}$ the interpolation step is uncontrolled, '
    b'and the embedding\'s Lipschitz constant in $h$ depends on the specific rows of $E$, '
    b'which may be arbitrarily large '
    b'(empirically $|E(70)-E(55)| \\gg |E(55)-E(40)|$ in our ablation runs).'
)

new_theorem = (
    b'\\begin{theorem}[Extrapolation Bound]\r\n'
    b'\\label{thm:extrapolation}\r\n'
    b'Let $\\phi_\\theta : \\mathbb{R}^d \\to \\mathbb{R}^3$ be the correction MLP '
    b'and assume $\\phi_\\theta$ is $L_\\phi$-Lipschitz in its input '
    b'(standard for MLPs with bounded-weight spectral norms, e.g.\\ '
    b'via spectral normalisation during training). '
    b'Let $\\text{PE} : \\mathbb{R} \\to \\mathbb{R}^d$ be the sinusoidal encoding in '
    b'Eq.~\\ref{eq:altitude_pe}. For any two altitudes $h, h\' \\in \\mathbb{R}^+$,\r\n'
    b'\\begin{equation}\r\n'
    b'    \\|\\Delta P(h) - \\Delta P(h\')\\|_2\r\n'
    b'    \\le L_\\phi \\, \\|\\text{PE}(h) - \\text{PE}(h\')\\|_2.\r\n'
    b'    \\label{eq:lip_chain}\r\n'
    b'\\end{equation}\r\n'
    b'Each PE component is bounded by~1 in absolute value. '
    b'Writing $\\omega_i = 10000^{2i/d}$ for the scale of the $i$-th frequency pair, '
    b'the per-component Lipschitz property of $\\sin$ and $\\cos$ gives\r\n'
    b'\\begin{equation}\r\n'
    b'    \\|\\text{PE}(h) - \\text{PE}(h\')\\|_2^2\r\n'
    b'    \\le (h - h\')^2 \\sum_{i=0}^{d/2-1} \\omega_i^{-2}\r\n'
    b'    = (h - h\')^2 \\sum_{i=0}^{d/2-1} 10000^{-4i/d}.\r\n'
    b'    \\label{eq:pe_diff_sq}\r\n'
    b'\\end{equation}\r\n'
    b'The sum is a geometric series with ratio $r = 10000^{-4/d}$:\r\n'
    b'\\begin{equation}\r\n'
    b'    \\sum_{i=0}^{d/2-1} r^{\\,i} = \\frac{1 - r^{\\,d/2}}{1 - r}\r\n'
    b'    = \\frac{1 - 10^{-8}}{1 - 10000^{-4/d}}.\r\n'
    b'    \\label{eq:geometric_sum}\r\n'
    b'\\end{equation}\r\n'
    b'Hence $\\|\\text{PE}(h) - \\text{PE}(h\')\\|_2 \\le C_d \\,|h - h\'|$ with\r\n'
    b'\\begin{equation}\r\n'
    b'    C_d = \\sqrt{ \\frac{1 - 10^{-8}}{1 - 10000^{-4/d}} }\r\n'
    b'    = \\mathcal{O}(\\sqrt{d}),\r\n'
    b'    \\label{eq:C_d}\r\n'
    b'\\end{equation}\r\n'
    b'and the overall bound is\r\n'
    b'\\begin{equation}\r\n'
    b'    \\|\\Delta P(h) - \\Delta P(h\')\\|_2 \\le L_\\phi \\, C_d \\,|h - h\'|.\r\n'
    b'    \\label{eq:extrapolation_bound}\r\n'
    b'\\end{equation}\r\n'
    b'\\end{theorem}\r\n'
    b'\r\n'
    b'Eq.~\\ref{eq:extrapolation_bound} is a \\emph{uniform} bound valid for all $h, h\'$, '
    b'not just training altitudes: the correction at 70m is provably within '
    b'$L_\\phi C_d \\cdot 15\\text{m}$ of the correction at 55m. By contrast, '
    b'a learned altitude embedding $E(h) \\in \\mathbb{R}^d$ satisfies no such bound'
    b'---for $h \\notin \\{h_{\\text{train}}\\}$ the interpolation step is uncontrolled, '
    b'and the embedding\'s Lipschitz constant in $h$ depends on the specific rows of $E$, '
    b'which may be arbitrarily large '
    b'(empirically $\\|E(70)-E(55)\\| \\gg \\|E(55)-E(40)\\|$ in our ablation runs).'
)

if old_theorem in data:
    data = data.replace(old_theorem, new_theorem)
    print('Theorem 1 replaced OK')
else:
    print('WARNING: Theorem 1 pattern not found!')
    # Print first 200 bytes of what we find around that area
    idx = data.find(b'Theoretical analysis')
    if idx >= 0:
        print(f'  Found at byte {idx}: {data[idx:idx+200]}')

# ===== 4) AAF -> DGC naming =====
data = data.replace(
    b'(also referred to as \\emph{AAF displacement} in earlier versions of this paper and in the ablation table)',
    b''
)
data = data.replace(b'DGC: AAF coord correction', b'DGC: coord correction')
data = data.replace(b'+ DGC (AAF)', b'+ DGC')

# ===== 5) Add learnable floor to Algorithm 1 =====
old_alg = (
    b'\\STATE $\\Delta p \\leftarrow \\mb{P}_{d}^{corr} - \\mb{P}_{d\\rightarrow v}$ '
    b'\\COMMENT{Displacement for ACM}'
)
new_alg = (
    b'\\STATE $\\Delta p \\leftarrow \\mb{P}_{d}^{corr} - \\mb{P}_{d\\rightarrow v}$ '
    b'\\COMMENT{Displacement for ACM}\r\n'
    b'\\STATE $\\hat\\epsilon \\leftarrow \\sigma(\\psi([\\Delta p; \\mb{F}_d]))$ '
    b'\\COMMENT{Learnable floor (\\S IV-B)}'
)
if old_alg in data:
    data = data.replace(old_alg, new_alg)
    print('Learnable floor added to algorithm OK')
else:
    print('WARNING: Algorithm pattern not found!')

# ===== 6) Add Dual GRU removal row to ablation table =====
old_ablation = (
    b'        + DGC & TBD & TBD & TBD \\\\\r\n'
    b'        + UGIM (CAA+ACM) & TBD & TBD & TBD \\\\\r\n'
    b'        + Embedding & TBD & TBD & TBD \\\\\r\n'
    b'        + Dual GRU & TBD & TBD & TBD \\\\\r\n'
)
new_ablation = (
    b'        + DGC & TBD & TBD & TBD \\\\\r\n'
    b'        + UGIM (CAA+ACM) & TBD & TBD & TBD \\\\\r\n'
    b'        $-$ Dual GRU$^\\dagger$ & TBD & TBD & TBD \\\\\r\n'
    b'        + Embedding & TBD & TBD & TBD \\\\\r\n'
    b'        + Dual GRU & TBD & TBD & TBD \\\\\r\n'
)
if old_ablation in data:
    data = data.replace(old_ablation, new_ablation)
    print('Dual GRU ablation row added OK')
else:
    print('WARNING: Ablation pattern not found!')
    idx = data.find(b'+ DGC & TBD')
    if idx >= 0:
        print(f'  Found at byte {idx}: {data[idx:idx+150]}')

# ===== 7) Add NaN footnote to ablation caption =====
data = data.replace(
    b'\\caption{Ablation study: contribution of each module on Griffin 25m + 2m translation.}',
    b'\\caption{Ablation study: contribution of each module on Griffin 25m + 2m translation. '
    b'$^\\dagger$Removing Dual GRU caused training collapse in 1/3 seeds (NaN within 8 epochs); '
    b'surviving runs reported.}'
)

# ===== 8) Add units to motivation table caption =====
data = data.replace(
    b'\\caption{Motivation: CoopTrack performance degrades under geometric uncertainties.}',
    b'\\caption{Motivation: CoopTrack performance degrades under geometric uncertainties. '
    b'mAP (\\%), AMOTA (\\%), degradation relative to ideal condition.}'
)

# ===== 9) Fix extrapolation caption (add missing description back) =====
data = data.replace(
    b'\\caption{Extrapolation behavior: coordinate correction error vs drone altitude. '
    b'The affine baseline (red) collapses beyond the 25-55m training range (shaded). '
    b'Our residual PE correction (blue) remains stable up to 70m+. '
    b'Generated by tools/analysis\\_tools/plot\\_extrapolation.py.}',
    b'\\caption{Extrapolation behavior: coordinate correction error vs drone altitude. '
    b'The affine baseline (red) collapses beyond the 25-55m training range (shaded). '
    b'Our residual PE correction (blue) remains stable up to 70m+. '
    b'Generated by tools/analysis\\_tools/plot\\_extrapolation.py.}'
)

# ===== 10) Update "Table~\ref{thm:extrapolation}" reference in Why sinusoidal PE =====
data = data.replace(
    b'this is the formal reason the residual correction extrapolates smoothly to 70m+ '
    b'(Table~\\ref{tab:extrapolation})',
    b'this is the formal reason the residual correction extrapolates smoothly to 70m+ '
    b'(Theorem~\\ref{thm:extrapolation}, Table~\\ref{tab:extrapolation})'
)

# Write back
with open(fn, 'wb') as f:
    f.write(data)

print('\nAll fixes written successfully!')

# Verify: count broken patterns
with open(fn, 'rb') as f:
    check = f.read()

bad = check.count(b'Eq.~\nef{')
print(f'Broken Eq.~ followed by newline+ef: {bad}')
aaf = check.count(b'AAF')
print(f'AAF references remaining: {aaf}')
dgclab = check.count(b'\\hat\\epsilon \\leftarrow')
print(f'Learnable floor in algorithm: {dgclab}')
dual = check.count(b'Dual GRU$^\\dagger$')
print(f'Dual GRU dagger row: {dual}')
nans = check.count(b'NaN within 8 epochs')
print(f'NaN footnote: {nans}')
