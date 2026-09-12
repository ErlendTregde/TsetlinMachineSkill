"""
Print the propositional formulas a trained pyTsetlinMachine has learned.

This is the whole point of using a Tsetlin Machine: the model is
"easy-to-interpret propositional formulas, composed by a collective of Tsetlin Automata"
[TM-README]. A script that only calls fit() and predict() throws that away.

Conventions encoded below, all from the pyTsetlinMachine interpretability demo
[PyTM-README, Interpretability Demo]:

  * tm.ta_action(class_index, clause_index, literal_index) returns 1 if that literal
    is INCLUDED in the clause.
  * literal_index k  <  n_features  ->  the plain feature  x_k
  * literal_index k  >= n_features  ->  the negated feature  not x_(k - n_features)
  * even clause indices (0, 2, 4, ...) are the POSITIVE clauses for the class
    (they vote FOR it); odd indices (1, 3, 5, ...) are the NEGATIVE clauses.

Sources: see ../references/sources.md
"""


def clause_to_string(tm, class_index, clause_index, n_features, feature_names=None):
    """Return one clause as a readable conjunction, or None if the clause is empty."""
    parts = []
    for k in range(n_features * 2):
        if tm.ta_action(class_index, clause_index, k) != 1:
            continue
        if k < n_features:
            name = feature_names[k] if feature_names else f"x{k}"
            parts.append(name)
        else:
            j = k - n_features
            name = feature_names[j] if feature_names else f"x{j}"
            parts.append(f"NOT {name}")
    if not parts:
        return None
    return " AND ".join(parts)


def show_clauses(tm, n_classes, n_clauses, n_features,
                 feature_names=None, max_per_group=10):
    """Print positive and negative clauses for every class.

    n_clauses is the clause count passed to the constructor.
    Pass feature_names so the output reads in the user's own vocabulary rather than x0, x1, ...
    """
    for c in range(n_classes):
        print(f"\n{'=' * 60}\nCLASS {c}\n{'=' * 60}")

        for label, indices in (
            ("Positive clauses (vote FOR this class)", range(0, n_clauses, 2)),
            ("Negative clauses (vote AGAINST this class)", range(1, n_clauses, 2)),
        ):
            print(f"\n{label}:")
            shown = 0
            empty = 0
            for j in indices:
                if shown >= max_per_group:
                    break
                text = clause_to_string(tm, c, j, n_features, feature_names)
                if text is None:
                    empty += 1
                    continue
                print(f"  Clause #{j}: {text}")
                shown += 1
            if shown == 0:
                print("  (none with included literals)")
            if empty:
                print(f"  ({empty} empty clause(s) skipped -- many empties can mean s is too low)")


def summarise_literal_usage(tm, class_index, n_clauses, n_features, feature_names=None, top=10):
    """Count how often each feature appears across a class's clauses.

    A quick global view of which features the model actually relies on.
    Note: this is a convenience summary, not a published interpretability method. [heuristic]
    For principled global/local interpretation see Blakely & Granmo, closed-form expressions
    for global and local interpretation of Tsetlin Machines (listed in TM-README's bibliography).
    """
    counts = {}
    for j in range(n_clauses):
        for k in range(n_features * 2):
            if tm.ta_action(class_index, j, k) != 1:
                continue
            base = k if k < n_features else k - n_features
            name = feature_names[base] if feature_names else f"x{base}"
            if k >= n_features:
                name = f"NOT {name}"
            counts[name] = counts.get(name, 0) + 1

    ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:top]
    print(f"\nMost used literals for class {class_index}:")
    for name, count in ranked:
        print(f"  {name}: in {count} clause(s)")
    return ranked


if __name__ == "__main__":
    print(__doc__)
    print("Import show_clauses(tm, n_classes, n_clauses, n_features, feature_names) "
          "after fitting a MultiClassTsetlinMachine.")
