# Three-arm ABMIL module-freezing intervention

Approved: Raw UNI1024 full bags, original721/251/saved fivefold, unchanged ABMIL/Cox32 merged tail/dropout.25/Adam1e-4 WD5e-4. No project unimodal checkpoint initialization; each scope starts newly at seed123.

Each Outer Train uses the existing event-stratified80/20 FIT/STOP seed123. Inner FIT common warmup3fixed Epochs, branch baseline/freeze_ab/freeze_c from exact last warmup state. Restore identical Python/NumPy/Torch CPU/CUDA RNG at branch start; new Adam only trainable parameters. All modules stay train() including frozen dropout. No no_grad around frozen C.

Predefine20 continuation Epochs per inner arm, no early stop; select best STOP C among continuation0..20, ties choose earlier. Candidate0 is common warmup checkpoint. Every arm receives same continuation budget and same stochastic/patient order prefix. Save eval Train/STOP metrics at each checkpoint with RNG preservation.

Outer full Train independently warms up3new Epochs, branches identically, trains each arm for its own inner-selected continuation budget. No Outer Val access while fitting/choosing budget. Final eval Train/Outer Val once per arm; offline eval C, Cox,36/60monthAUC/KM as existing implementation. Frozen-state exact equality and frozen gradients checked; branch init hash/RNG and batch-order hashes audited. Source model topology and main training unchanged.

Tests: freezing scopes/parameter counts, C-to-AB gradient flow, dropout mode, common branch init and RNG, exact frozen tensors. CPU miniature end-to-end plus realGPU smoke before full source-pinned background launcher. Five fold jobs max3concurrent. All models/IDs/predictions stay private, aggregate report inside this directory. No automatic push unless requested. This tests freezing interventions, not unique causal attribution to A/B/C; repeated cohort exploration and single inner STOP holdout remain limitations.
