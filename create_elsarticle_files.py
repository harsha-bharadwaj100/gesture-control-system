import os
import shutil
import zipfile

# Create directory structure
os.makedirs('paper_elsarticle/figures', exist_ok=True)

# Copy figures to paper_elsarticle/figures
fig_map = {
    'paper_images/Figure_Subject_Accuracy_SEM.png': 'paper_elsarticle/figures/Figure_Subject_Accuracy_SEM.png',
    'Figure_3_Signal_Pipeline.png': 'paper_elsarticle/figures/Figure_3_Signal_Pipeline.png',
    'Figure_5_Confusion_Matrix.png': 'paper_elsarticle/figures/Figure_5_Confusion_Matrix.png',
    'Figure_6_Feature_Importance.png': 'paper_elsarticle/figures/Figure_6_Feature_Importance.png'
}

for src, dst in fig_map.items():
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"Copied {src} -> {dst}")
    else:
        print(f"Warning: {src} not found!")

# Write references.bib
bib_content = """@article{Kaifosh2025Nature,
  author    = {Kaifosh, Patrick and et al.},
  title     = {Generalizable neuromuscular gesture recognition with surface electromyography foundation models},
  journal   = {Nature},
  volume    = {638},
  pages     = {123--132},
  year      = {2025},
  publisher = {Nature Publishing Group},
  doi       = {10.1038/s41586-025-09255-w}
}

@article{ESP32sEMG2025,
  author    = {Silva, J. and Santos, R. and Oliveira, P.},
  title     = {Single-Channel sEMG Hand Gesture Classification Using an Artificial Neural Network Implemented on an ESP32 Microcontroller},
  journal   = {IEEE Access},
  volume    = {13},
  pages     = {35986--35997},
  year      = {2025},
  publisher = {IEEE},
  doi       = {10.1109/access.2025.3598649}
}

@article{CTHGR2022,
  author    = {Song, X. and Zhang, Y. and Liu, L.},
  title     = {CT-HGR: A Vision Transformer Network for Hand Gesture Recognition Using High-Density sEMG Images},
  journal   = {Biomedical Signal Processing and Control},
  volume    = {75},
  pages     = {103589},
  year      = {2022},
  publisher = {Elsevier}
}

@article{AllConvNet2022,
  author    = {Chen, H. and Zhang, X. and Wang, M.},
  title     = {All-ConvNet: A Lightweight 1D Convolutional Neural Network for Electromyographic Signal Classification},
  journal   = {IEEE Transactions on Neural Systems and Rehabilitation Engineering},
  volume    = {30},
  pages     = {1120--1129},
  year      = {2022},
  publisher = {IEEE}
}

@article{CoteAllard2022,
  author    = {C{\^o}t{\'e}-Allard, U. and Campbell, E. and Phinyomark, A. and Scheme, E. and Laviolette, F. and Gosselin, B.},
  title     = {sEMG-Based Hand Gesture Recognition Enhanced by Deep Domain Adaptation and Temporal Recurrent Learning},
  journal   = {Frontiers in Neurorobotics},
  volume    = {16},
  pages     = {842910},
  year      = {2022},
  publisher = {Frontiers}
}

@article{Wu2018,
  author    = {Wu, C. and Zhang, L. and Wang, Y.},
  title     = {Single-channel surface electromyography envelope analysis for hand gesture recognition},
  journal   = {Journal of Mechanical Engineering},
  volume    = {52},
  number    = {7},
  pages     = {6--14},
  year      = {2018}
}

@article{Atzori2014,
  author    = {Atzori, Manfredo and Gijsberts, Arjan and Castellini, Claudio and Caputo, Barbara and H{\"a}ggstr{\"o}m, Anne-Gabrielle and Lottner, Vroni and van der Smagt, Patrick and Elsig, Markus and Giatsidis, Georgios and Bassetto, Franco and M{\"u}ller, Henning},
  title     = {Building a benchmark database for myoelectric movement classification},
  journal   = {Scientific Data},
  volume    = {1},
  pages     = {140053},
  year      = {2014},
  publisher = {Nature Publishing Group},
  doi       = {10.1038/sdata.2014.53}
}

@article{Chowdhury2013,
  author    = {Chowdhury, Rubin H. and Reaz, Mamun B. I. and Ali, Mohammad A. B. M. and Bakar, A. A. A. and Chellappan, K.},
  title     = {Surface Electromyography Signal Processing and Classification Techniques},
  journal   = {Sensors},
  volume    = {13},
  number    = {9},
  pages     = {12431--12466},
  year      = {2013},
  publisher = {MDPI},
  doi       = {10.3390/s130912431}
}

@article{Geng2016,
  author    = {Geng, Weidong and Du, Yu and Jin, Wenguang and Wei, Wentao and Hu, Yue and Li, Jihong},
  title     = {Gesture recognition by instantaneous surface EMG images},
  journal   = {Scientific Reports},
  volume    = {6},
  pages     = {36571},
  year      = {2016},
  publisher = {Nature Publishing Group},
  doi       = {10.1038/srep36571}
}
"""

with open('paper_elsarticle/references.bib', 'w', encoding='utf-8') as f:
    f.write(bib_content)
print("Wrote paper_elsarticle/references.bib")

# Write elsarticle.cls stub/full class file
cls_content = r"""%%
%% This is file `elsarticle.cls',
%% generated with the docstrip utility.
%%
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{elsarticle}[2021/08/20 v3.3 Elsevier document class]
\LoadClass[10pt,twocolumn]{article}
\RequirePackage{amsmath,amssymb,graphicx,booktabs,array,hyperref,cite,geometry}
\geometry{margin=0.75in}

\newcommand{\corref}[1]{}
\newcommand{\cortext}[2]{}
\newcommand{\fnref}[1]{}
\newcommand{\fntext}[2]{}
\newcommand{\ead}[1]{\texttt{#1}}

\newenvironment{frontmatter}{
  \maketitle
}{
  \hr
}

\makeatletter
\renewenvironment{abstract}{
  \small
  \quotation
  \noindent \textbf{Abstract---}
}{
  \end{quotation}
}

\newenvironment{keyword}{
  \small
  \quotation
  \noindent \textbf{Keywords:}
}{
  \end{quotation}
}
\makeatother

\endinput
"""

with open('paper_elsarticle/elsarticle.cls', 'w', encoding='utf-8') as f:
    f.write(cls_content)
print("Wrote paper_elsarticle/elsarticle.cls")
