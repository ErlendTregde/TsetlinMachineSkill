# TsetlinMachineSkill

A Claude Code skill for writing, debugging and explaining **Tsetlin Machine** code: TMU,
pyTsetlinMachine and GraphTsetlinMachine.

Tsetlin Machines learn human-readable logic rules (clauses) instead of opaque weights. That makes
them a good fit when you have to explain a model to someone, but the ecosystem is fragmented and
general-purpose assistants tend to get the APIs wrong. This skill gives Claude a sourced, tested
playbook for doing it properly.

## Install (Claude Code)

```
/plugin marketplace add ErlendTregde/TsetlinMachineSkill
/plugin install tsetlin-machine@tsetlin-machine-skill
```

After installing, Claude loads the skill automatically whenever Tsetlin Machines, TMU, GraphTM,
clauses or interpretable/logic-based ML come up. You can also call it directly with
`/tsetlin-machine:tsetlin-machine`.

## How it works

The skill walks Claude through a fixed five-step workflow instead of letting it improvise:

1. **Pick the right library before writing any code.** There are more than fifteen TM
   implementations, and their APIs differ in ways that fail silently. The skill routes to:
   - **TMU** (the default) for tabular, image or most other new work
   - **GraphTsetlinMachine** for graphs, sequences and variable-sized input
   - **pyTsetlinMachine** only for existing code or reproducing older papers
   - a **from-scratch** implementation only when no library will install, or when you want to learn
     how a TM works
   It also carries install notes tested on Windows and Linux, for example that TMU must be installed
   from git because the PyPI release breaks on numpy 2.
2. **Run a smoke test first.** Run the library's Noisy XOR demo before touching your data, so you
   can tell an install problem from a modelling problem. TMU's typical failure is a silent hang
   rather than an error.
3. **Booleanize explicitly.** TMs take Boolean input. The skill converts continuous features with a
   thermometer encoding fitted on the training set only, and prints the feature count and an
   example row so you can see what it did.
4. **Set T and s deliberately.** These two hyperparameters matter most. Claude states its starting
   values and offers a small search, rather than presenting one run as the result.
5. **Always extract the clauses.** Interpretability is the reason to use a TM, so every script
   prints the learned rules using your own feature names and explains the most important ones in
   plain language.

The skill also has a **diagnostic table** that maps common symptoms (accuracy plateaus, clauses too
specific, GraphTM stuck at chance level) to their cause and a fix.

### Every claim has a source

Each factual statement in the skill has a tag such as `[TM-README §Learning]` or
`[arXiv:1909.07310]`, and the tags resolve to URLs in `references/sources.md`. Statements marked
`[heuristic]` have no source and are presented to you as suggestions to test.
`[verified-YYYY-MM-DD]` means the statement was checked by actually running the code on that date.

### Context is loaded only when needed

`SKILL.md` holds the workflow. The detailed material for each library sits in separate reference
files, and Claude reads only the one for the library it chose, so the rest doesn't take up context.

## Structure

```
.claude-plugin/
  plugin.json             # plugin manifest (name, version, author)
  marketplace.json        # lets people install this repo with /plugin
skills/tsetlin-machine/
  SKILL.md                # entry point: triggers, the 5-step workflow, diagnostic table
  references/
    concepts.md           # how a TM classifies and learns (for explanations)
    from-scratch.md       # the exact algorithm, checked against CAIR's reference code
    tmu.md                # TMU: install, API, clause extraction
    pytsetlinmachine.md   # pyTsetlinMachine: API, published configs, clause extraction
    graphtm.md            # GraphTM: graph construction, symbols, depth, CUDA
    booleanization.md     # turning real-valued data into Boolean features
    sources.md            # every citation tag resolved to a URL
  scripts/
    smoke_test.py         # Noisy XOR check: python smoke_test.py tmu|pytm|graphtm
    booleanize.py         # thermometer encoder with a describe() report
    show_clauses.py       # prints learned clauses from pyTsetlinMachine
  evals/
    evals.json            # test prompts with assertions used to check the skill
```

## Evals

`evals/evals.json` contains realistic user prompts, such as building an explainable readmission
model from a CSV of patient data, with hidden ground-truth rules and assertions. They check whether
Claude picks a real library, searches T and s, and recovers the true rules in the printed clauses.

## License

MIT
