# Project research constraints

- The proposed method is a multimodal pathology–CT teacher and a CT student.
  Project-trained unimodal pathology/CT models are comparison baselines only.
- Do not initialize, warm-start, or stage teacher/student training from this
  project's trained unimodal checkpoints. Do not propose frozen unimodal
  encoders as a replacement method. Revisit this only if the user explicitly
  changes the constraint.
- Continue using the separately configured public pretrained encoders and
  fixed foundation-model patch features where specified. This is distinct
  from loading the project's downstream-trained unimodal checkpoints.
- Train the teacher's pathology and CT branches jointly according to the
  experiment settings; the student receives the specified teacher distillation
  supervision. Public-pretrained/newly initialized modules and teacher
  distillation do not authorize loading a trained unimodal baseline.
- Historical diagnostic experiments may be retained, but their frozen or
  unimodal-initialized designs must not be presented as the proposed method.
