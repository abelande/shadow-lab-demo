"""
p6lab.patterns — Pattern Library System (Spec §5)

- library.py: YAML registry + Pydantic models (§5.1)
- miner.py: HDBSCAN unsupervised discovery (§5.2)
- labeler.py: Forward-outcome labeling (§5.3)
- template_matcher.py: Cosine similarity ensemble scoring (§5.4)

Import from submodules directly — this package stays lazy because some
submodules (template_matcher) still contain NotImplementedError stubs
until Phase 4:
    from p6lab.patterns.library import PatternLibrary, PatternDefinition
    from p6lab.patterns.miner import mine
"""
