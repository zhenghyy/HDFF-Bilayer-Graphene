# Bundled model architecture

The four checkpoints share the same HDFF graph architecture. Only the target
normalization and fitted parameters (and, for some trials, dropout) differ.

1. Parse an **initial** bilayer graphene POSCAR. Carbon forms the host lattice;
   the non-carbon atoms are dopant sites. No optimized coordinates enter the
   descriptor or model.
2. Build a local, distance-defined five-shell neighborhood around each dopant.
   Each local window carries atom-property, element, center/dopant indicator,
   shell, and geometric information. The fitted preprocessing statistics are
   bundled with each checkpoint.
3. Select 20 atomic properties from the source property table and eight
   structure-level values. The selected-property indices are stored in the
   checkpoint. The property-only representation excludes a learned element-ID
   embedding.
4. Encode local dopant neighborhoods, then perform eight global interaction
   layers among dopant sites. Layer relations and distances enter the global
   branch. Gated exchanges pass information between local and global branches.
   The selected variant uses no learned edge-role embedding (`role_dim=0`).
5. Fuse the local, global, and structure-level representations into one
   normalized regression output. The checkpoint's target mean and standard
   deviation convert it back to the property unit (eV or Å).

Implementation: `hdff_model_portable/training_workspace/models/hdff_scalable.py`,
`hdff_element_study_v1.py`, and `hdff_embedding_ablation_v1.py`. Descriptor
construction is in `dopant_graph/featurizer.py` and related modules. This is
the inference framework required to load `best.pt`; training controllers and
raw DFTB+ processing files are intentionally excluded.
