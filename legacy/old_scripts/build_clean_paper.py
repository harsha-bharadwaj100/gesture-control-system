# -*- coding: utf-8 -*-
import os, re, shutil, zipfile

with open('paper_elsarticle/main.tex', 'r', encoding='utf-8') as f:
    orig = f.read()

# Extract the 27 LSTM and 27 AE rows from the original file
lstm_rows = []
ae_rows = []
for line in orig.splitlines():
    line_s = line.strip()
    if line_s.startswith('1D-CNN + LSTM &'):
        parts = [p.strip() for p in line_s.rstrip(r'\\').split('&')]
        opt = parts[1]
        lr = parts[2]
        batch = parts[3].replace('Batch=', '')
        epoch = parts[4]
        acc = parts[5]
        time = parts[6].rstrip('s')
        lstm_rows.append((opt, lr, batch, epoch, acc, time))
    elif line_s.startswith('1D-CNN + Autoencoder &'):
        parts = [p.strip() for p in line_s.rstrip(r'\\').split('&')]
        opt = parts[1]
        lr = parts[2]
        latent = parts[3].replace('LatentDim=', '')
        epoch = parts[4]
        acc = parts[5]
        time = parts[6].rstrip('s')
        ae_rows.append((opt, lr, latent, epoch, acc, time))

assert len(lstm_rows) == 27
assert len(ae_rows) == 27

# Build Table 4 side-by-side
table4_lines = [
    r'\begin{table*}[!t]',
    r'\centering',
    r'\footnotesize',
    r'\caption{Deep Learning Sensitivity Grid \& Convergence Analysis (Side-by-Side 54-Configuration Architecture Comparison)}',
    r'\label{tab:sensitivity_grid_full}',
    r'\resizebox{\textwidth}{!}{%',
    r'\begin{tabular}{lccccc|lccccc}',
    r'\toprule',
    r'\multicolumn{6}{c}{\textbf{1D-CNN + LSTM Network}} & \multicolumn{6}{c}{\textbf{1D-CNN + Autoencoder Network}} \\',
    r'\cmidrule(lr){1-6} \cmidrule(lr){7-12}',
    r'\textbf{Opt.} & \textbf{LR} & \textbf{Batch} & \textbf{Epoch} & \textbf{Val (\%)} & \textbf{Time (s)} & \textbf{Opt.} & \textbf{LR} & \textbf{Latent} & \textbf{Epoch} & \textbf{Val (\%)} & \textbf{Time (s)} \\',
    r'\midrule'
]

for (l_opt, l_lr, l_b, l_ep, l_acc, l_t), (a_opt, a_lr, a_lat, a_ep, a_acc, a_t) in zip(lstm_rows, ae_rows):
    row_str = f'{l_opt} & {l_lr} & {l_b} & {l_ep} & {l_acc} & {l_t}s & {a_opt} & {a_lr} & {a_lat} & {a_ep} & {a_acc} & {a_t}s \\\\'
    table4_lines.append(row_str)

table4_lines.extend([
    r'\bottomrule',
    r'\end{tabular}%',
    r'}',
    r'\end{table*}'
])
table4_latex = '\n'.join(table4_lines)

print('Table 4 generated successfully, length:', len(table4_latex))
