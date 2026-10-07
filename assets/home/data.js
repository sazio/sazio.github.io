// All site content lives here. Edit this file; the page renders from it.
window.SITE = {
  links: {
    email: "simone.azeglio@gmail.com",
    scholar: "https://scholar.google.com/citations?user=ld9Bs6oAAAAJ&hl=en",
    github: "https://github.com/sazio",
    x: "https://x.com/simoneazeglio",
    bluesky: "https://bsky.app/profile/s-azeglio.bsky.social",
    linkedin: "https://www.linkedin.com/in/simoneazeglio",
    cv: "/files/cv.pdf",
  },

  pillars: [
    {
      key: "statistics",
      name: "Statistics",
      question: "What lies beyond pairwise correlations?",
      title: "Higher-order structure in natural scenes",
      body:
        "Natural images carry structure that pairwise statistics miss, and retinal circuits pick it up through multiplicative interactions between their inputs. Building the same operation into convolutional layers improves image classification and gives better models of retinal responses than standard CNNs.",
      venues: "NeurIPS 2025 · arXiv 2025",
      links: [
        { label: "Higher-order convolution", url: "https://arxiv.org/abs/2412.06740" },
        { label: "Retina", url: "https://arxiv.org/abs/2505.07620" },
      ],
    },
    {
      key: "symmetry",
      name: "Symmetry",
      question: "What should a neural code leave unchanged?",
      title: "Equivariance as an inductive bias",
      body:
        "Prey grows on the retina as a mouse closes in. Specific ganglion cells (OFF-α) encode it in a scale-equivariant way, and scale-steerable networks match CNN predictivity with 84% fewer parameters. I am now extending the idea from scale to velocity.",
      venues: "CoSyNe 2026 · Mouse vs AI, NeurIPS 2025 (3rd place)",
      links: [
        { label: "CoSyNe thread", url: "https://x.com/simoneazeglio/status/2032039903945449961" },
      ],
    },
    {
      key: "information",
      name: "Information",
      question: "Which bits, about what?",
      title: "What neural information is about",
      body:
        "Mutual information says how much a population encodes, not what it is about. Starting from a few axioms, we split it across individual stimuli and features, and derive a unique multi-scale geometry on stimulus space that stretches well-encoded directions and is tied exactly to the mutual information. Diffusion models make both work on natural images.",
      venues: "NeurIPS 2026 Oral · NeurIPS 2025 Spotlight",
      links: [
        { label: "Multi-scale geometry · Oral", url: "https://arxiv.org/abs/2605.06304" },
        { label: "Decomposition · Spotlight", url: "https://arxiv.org/abs/2505.11309" },
        { label: "Score-based metric", url: "https://arxiv.org/abs/2505.11128" },
      ],
    },
  ],

  // selected: shown by default. tag: one of the pillar keys, or "other".
  publications: [
    {
      year: 2026, selected: true, tag: "information", highlight: "Oral",
      title: "A multi-scale information geometry reveals the structure of mutual information in neural populations",
      authors: "S. Azeglio, S. Laquitaine, U. Ferrari, M. Chalk",
      venue: "NeurIPS 2026",
      links: [{ label: "arXiv", url: "https://arxiv.org/abs/2605.06304" }],
    },
    {
      year: 2026, selected: false, tag: "other",
      title: "Scene structure predicts perceptual decisions in naturalistic detection tasks",
      authors: "J. Yang, T. Vercillo, T. E. Cutrona, S. Azeglio, G. Iannetti, P. Neri",
      venue: "bioRxiv",
    },
    {
      year: 2025, selected: true, tag: "information", highlight: "Spotlight · top 3%",
      title: "Decomposing stimulus-specific sensory neural information via diffusion models",
      authors: "S. Laquitaine*, S. Azeglio*, C. Paris, U. Ferrari, M. Chalk",
      venue: "NeurIPS 2025",
      links: [{ label: "arXiv", url: "https://arxiv.org/abs/2505.11309" }],
    },
    {
      year: 2025, selected: true, tag: "statistics",
      title: "Convolution goes higher-order: a biologically inspired mechanism empowers image classification",
      authors: "S. Azeglio, O. Marre, P. Neri, U. Ferrari",
      venue: "NeurIPS 2025",
      links: [{ label: "arXiv", url: "https://arxiv.org/abs/2412.06740" }],
    },
    {
      year: 2025, selected: true, tag: "information",
      title: "What's inside your diffusion model? A score-based Riemannian metric to explore the data manifold",
      authors: "S. Azeglio, A. Di Bernardo",
      venue: "arXiv",
      links: [{ label: "arXiv", url: "https://arxiv.org/abs/2505.11128" }],
    },
    {
      year: 2025, selected: true, tag: "statistics",
      title: "Higher-order convolution improves neural predictivity in the retina",
      authors: "S. Azeglio, V. C. Garcia, G. Glaziou, P. Neri, O. Marre, U. Ferrari",
      venue: "arXiv",
      links: [{ label: "arXiv", url: "https://arxiv.org/abs/2505.07620" }],
    },
    {
      year: 2023, selected: false, tag: "other",
      title: "Retrospective on the SENSORIUM 2022 competition",
      authors: "K. F. Willeke, …, S. Azeglio, U. Ferrari, P. Neri, O. Marre, …, F. Sinz",
      venue: "PMLR",
    },
    {
      year: 2022, selected: false, tag: "symmetry",
      title: "Symmetry and geometry in neural representations",
      authors: "S. Sanborn, C. Shewmake, S. Azeglio, A. Di Bernardo, N. Miolane",
      venue: "PMLR (NeurReps proceedings)",
      links: [{ label: "PMLR", url: "http://proceedings.mlr.press/v197/" }],
    },
    {
      year: 2022, selected: false, tag: "other",
      title: "Improving neural predictivity in the visual cortex with gated recurrent connections",
      authors: "S. Azeglio, S. Poetto, L. Savant Aira, M. Nurisso",
      venue: "Brain-Score Workshop, CoSyNe 2022",
      links: [{ label: "PDF", url: "https://openreview.net/references/pdf?id=HbNa-jRWf5" }],
    },
    {
      year: 2021, selected: false, tag: "other",
      title: "Topological data analysis techniques enhance hand pose classification from ECoG neural recordings",
      authors: "S. Azeglio, A. Di Bernardo, G. Penna, F. Pittatore, S. Poetto, J. Gruenwald, C. Kapeller, K. Kamada, C. Guger",
      venue: "arXiv",
    },
    {
      year: 2021, selected: false, tag: "other",
      title: "NeuralPDE: automating physics-informed neural networks (PINNs) with error approximations",
      authors: "K. Zubov, Z. McCarthy, Y. Ma, F. Calisto, V. Pagliarino, S. Azeglio, …, C. Rackauckas",
      venue: "arXiv",
    },
    {
      year: 2021, selected: false, tag: "other",
      title: "Physics-informed machine learning simulator for wildfire propagation",
      authors: "L. Bottero, F. Calisto, G. Graziano, V. Pagliarino, M. Scauda, S. Tiengo, S. Azeglio",
      venue: "AAAI-MLPS 2021",
    },
  ],

  workshops: [
    { year: "2026", name: "Efficient Coding in the Modern Age: Adaptive Representations for Vision across Brains and Machines", venue: "CoSyNe" },
    { year: "2025", name: "Symmetry and Geometry in Neural Representations (NeurReps), 4th ed.", venue: "NeurIPS", url: "https://proceedings.mlr.press/v282/" },
    { year: "2024", name: "Symmetry and Geometry in Neural Representations (NeurReps), 3rd ed.", venue: "NeurIPS", url: "https://proceedings.mlr.press/v282/" },
    { year: "2024", name: "Sharpening our Sight: Naturalistic Visual Perception through Efficient Representations and Active Search", venue: "CoSyNe", url: "https://sazio.github.io/workshops/cosyne2024/" },
    { year: "2023", name: "Symmetry and Geometry in Neural Representations (NeurReps), 2nd ed.", venue: "NeurIPS", url: "https://sazio.github.io/workshops/neurreps/" },
    { year: "2023", name: "Symmetry, Invariance and Neural Representations, 2nd ed.", venue: "Bernstein", url: "https://sazio.github.io/workshops/sinr/" },
    { year: "2022", name: "Symmetry and Geometry in Neural Representations (NeurReps)", venue: "NeurIPS", url: "https://sazio.github.io/workshops/neurreps2022/" },
    { year: "2022", name: "Symmetry, Invariance and Neural Representations", venue: "Bernstein", url: "https://sazio.github.io/workshops/sinr2022/" },
  ],

  news: [
    { date: "2026-09-04", text: "NeurReps 2024 and 2025 proceedings published as PMLR volume 282, which I co-edited", url: "https://proceedings.mlr.press/v282/" },
    { date: "2026-07-01", text: "Defended my PhD thesis at the Vision Institute (Sorbonne) and ENS Paris" },
    { date: "2026-03-14", text: "Co-organizing a workshop on modern efficient coding at CoSyNe 2026", url: "https://x.com/simoneazeglio/status/2032906430097883334" },
    { date: "2026-03-12", text: "CoSyNe poster 1-146: mouse OFF-α ganglion cells are scale equivariant", url: "https://x.com/simoneazeglio/status/2032039903945449961" },
    { date: "2025-12-10", text: "NeurReps 2025, our 4th edition, is a wrap", url: "https://x.com/simoneazeglio/status/1998808925509157012" },
    { date: "2025-12-07", text: "3rd place in the Mouse vs AI competition at NeurIPS, with scale-equivariant tracking agents", url: "https://x.com/simoneazeglio/status/1997753182035087686" },
    { date: "2025-12-01", text: "Diffusion-model decomposition of neural information is a NeurIPS 2025 Spotlight", url: "https://x.com/simoneazeglio/status/1995477602602004760" },
    { date: "2025-12-01", text: "Convolution goes higher-order at NeurIPS 2025", url: "https://x.com/simoneazeglio/status/1995470640271262137" },
    { date: "2025-08-14", text: "Announcing the NeurReps 2025 Prize", url: "https://x.com/simoneazeglio/status/1955986743393407455" },
    { date: "2025-07-10", text: "NeurReps returns to NeurIPS for a 4th edition", url: "https://x.com/simoneazeglio/status/1943298116469334513" },
    { date: "2024-12-14", text: "Opened NeurReps 2024 in Vancouver", url: "https://proceedings.mlr.press/v282/" },
    { date: "2024-07-03", text: "Launched the NeurReps worldwide seminar series (MILA, UPenn, MIT, Harvard, Amsterdam)", url: "https://www.youtube.com/@neurreps" },
    { date: "2024-01-30", text: "Sharpening our Sight workshop at CoSyNe 2024", url: "https://sites.google.com/view/cosyne2024-sos/home" },
    { date: "2023-12-11", text: "NeurReps 2023 at NeurIPS", url: "https://sazio.github.io/workshops/neurreps/" },
    { date: "2023-05-30", text: "Art and AI: video representations, with artist Kaspar Ravel at Sorbonne", url: "https://www.youtube.com/watch?v=2qsx2bEztho" },
    { date: "2022-12-06", text: "3rd place in the SENSORIUM competition at NeurIPS 2022", url: "https://sensorium2022.net/home" },
    { date: "2022-12-02", text: "First NeurReps workshop live at NeurIPS 2022", url: "https://sazio.github.io/workshops/neurreps2022/" },
    { date: "2022-06-13", text: "Brains for Brains Young Researcher Award, Bernstein Network", url: "https://bernstein-network.de/en/newsroom/news/brains-for-brains-awardee-2022/" },
  ],

  writing: [
    { year: 2020, title: "A bunch of words: an introduction to spaCy on CORD-19", url: "https://sazio.github.io/posts/2020/10/A-Bunch-of-Words:-an-Introduction-to-SpaCy-on-CORD-1/" },
    { year: 2020, title: "A second step into feature engineering: feature selection", url: "https://sazio.github.io/posts/2020/08/A-Second-Step-into-Feature-Engineering:Feature-Selection/" },
    { year: 2020, title: "An introduction to feature engineering: feature importance", url: "https://sazio.github.io/posts/2020/08/An-Introduction-to-Feature-Engineering:-Feature-Importance/" },
    { year: 2020, title: "Data spectrometry, or how to preprocess your data", url: "https://sazio.github.io/posts/2020/07/Data-Spectrometry-or-How-to-Preprocess-your-Data/" },
    { year: 2020, title: "Deep Fake Challenge: an overview", url: "https://sazio.github.io/posts/2020/06/Deep-Fake-Challenge:-an-Overview/" },
    { year: 2020, title: "The learning problem: comparing brain and machine", url: "https://sazio.github.io/posts/2020/05/The-Learning-Problem:-Comparison-between-Brain-and-Machine/" },
  ],
};
