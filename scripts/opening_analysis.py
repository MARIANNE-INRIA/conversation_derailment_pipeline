"""Opening-exchange outcome associations and figure-inspired A/B/C plots.

No model errors or future-message features enter this analysis. Attacker identity,
if supplied, is used only for retrospective subgrouping of derailed conversations.
"""
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact

PURPLE, GREEN = '#8054a5', '#2b9665'


def add_opening_arguments(parser):
    parser.add_argument('--opening_analysis', action='store_true', help='Also analyze the first two comments against actual conversation outcomes')
    parser.add_argument('--opening_only', action='store_true', help='Only the opening analysis; labels required but no model predictions/threshold needed')
    parser.add_argument('--opening_features', nargs='+', default=['annot1', 'annot2', 'politeness_strategy', 'prompt_type'], help='Categorical/multilabel columns, kept separate')
    parser.add_argument('--opening_binary_features', nargs='*', default=[], help='Boolean presence columns, e.g. gratitude direct_question')
    parser.add_argument('--opening_author_col', default='author')
    parser.add_argument('--opening_attacker_col', default='attacker_id', help='Actual eventual attacker identifier, never inferred from aggressive')
    parser.add_argument('--opening_metadata', help='Optional CSV with one conversation_id and attacker ID per conversation')
    parser.add_argument('--opening_test', choices=['binomial', 'fisher'], default='binomial', help='Binomial uses observed on-track feature prevalence as a plug-in null; Fisher is also exported')
    parser.add_argument('--opening_significance', choices=['raw', 'bh'], default='raw', help='p or BH q controls plot markers; both are exported')
    parser.add_argument('--opening_effect_threshold', type=float, default=.2)
    parser.add_argument('--opening_pseudocount', type=float, default=.5, help='Added to all four cells for log-odds and approximate 95%% CI, not tests')
    parser.add_argument('--opening_min_count', type=int, default=5, help='Minimum feature occurrences (awry + on-track) per comparison')
    parser.add_argument('--opening_max_features', type=int, default=30, help='Maximum plotted rows; all results are exported')


def feature_set(value, binary, separator):
    value = str(value).strip()
    if not value:
        return None
    if binary:
        if value.casefold() in ('1', '1.0', 'true', 'yes', 'oui'):
            return frozenset(['present'])
        if value.casefold() in ('0', '0.0', 'false', 'no', 'non'):
            return frozenset()
        raise ValueError(f'Invalid binary opening feature value: {value!r}')
    if value.startswith('['):
        try:
            items = json.loads(value)
            if isinstance(items, list) and all(isinstance(x, (str, int)) for x in items):
                return frozenset(str(x).strip().casefold() for x in items if str(x).strip())
        except json.JSONDecodeError:
            pass
    return frozenset(s.strip().casefold() for s in re.split(separator, value) if s.strip())


def significance_marks(p, symbol):
    if not np.isfinite(p) or p >= .05:
        return ''
    return symbol * (3 if p < .001 else 2 if p < .01 else 1)


def effect(a, n_awry, c, n_track, pseudocount=.5):
    """Rows: awry/on-track; columns: feature present/absent. Natural log OR."""
    if n_awry == 0 or n_track == 0:
        return dict(log_odds=np.nan, ci_low=np.nan, ci_high=np.nan, binomial_p=np.nan, fisher_p=np.nan,
                    reference_rate=np.nan)
    b, d = n_awry-a, n_track-c
    cells = np.array([a,b,c,d], dtype=float) + pseudocount
    log_odds = float(np.log(cells[0]) + np.log(cells[3]) - np.log(cells[1]) - np.log(cells[2]))
    se = float(np.sqrt((1/cells).sum()))
    reference = c/n_track
    # Explicit one-sample plug-in test; not a two-sample exact test and not a
    # paired test. Export Fisher alongside it so the choice is reviewable.
    return dict(log_odds=log_odds, ci_low=log_odds-1.96*se, ci_high=log_odds+1.96*se,
                binomial_p=float(binomtest(a, n_awry, reference, alternative='two-sided').pvalue),
                fisher_p=float(fisher_exact([[a,b],[c,d]], alternative='two-sided')[1]),
                reference_rate=reference)


def opening_cohort(joined, args):
    early = joined.loc[(joined.timestep <= 2) & (joined['_merge'] == 'both')].copy()
    eligible = early.groupby('conversation_id').timestep.agg(set)
    eligible = eligible[eligible.map(lambda x: x == {1,2})].index
    early = early.loc[early.conversation_id.isin(eligible)].drop(columns='_merge').copy().fillna('')
    if early.empty:
        raise ValueError('No conversations have annotations for both opening comments')
    first = early.loc[early.timestep == 1].set_index('conversation_id')
    second = early.loc[early.timestep == 2].set_index('conversation_id').reindex(first.index)
    metadata = None
    if args.opening_metadata:
        metadata = pd.read_csv(args.opening_metadata, dtype=str, keep_default_na=False, sep=args.sep)
        for col in ['conversation_id', args.opening_attacker_col]:
            if col not in metadata:
                raise ValueError(f'Missing opening metadata column: {col}')
            metadata[col] = metadata[col].str.strip()
        if metadata.conversation_id.eq('').any() or metadata.conversation_id.duplicated().any():
            raise ValueError('Opening metadata requires unique nonempty conversation_id values')
        metadata = metadata.set_index('conversation_id')
    cohort = pd.DataFrame({'label': first.label.astype(int)}, index=first.index)
    if set(cohort.label) != {0, 1}:
        raise ValueError('Opening analysis requires both awry and on-track conversations with two annotated opening comments')
    cohort['role_group'] = 'on_track'
    cohort.loc[cohort.label == 1, 'role_group'] = 'missing_attacker_or_author'
    for cid in cohort.index[cohort.label == 1]:
        if metadata is not None:
            attacker = metadata.at[cid, args.opening_attacker_col] if cid in metadata.index else ''
        elif args.opening_attacker_col in early:
            values = early.loc[early.conversation_id == cid, args.opening_attacker_col].astype(str).str.strip()
            known = set(values[values != ''])
            if len(known) > 1:
                raise ValueError(f'Conflicting attacker identities for {cid}')
            attacker = next(iter(known), '')
        else:
            attacker = ''
        if not attacker or args.opening_author_col not in early:
            continue
        author1 = str(first.at[cid, args.opening_author_col]).strip()
        author2 = str(second.at[cid, args.opening_author_col]).strip()
        if not author1 or not author2:
            continue
        if author1 == author2:
            group = 'same_author_in_both_openings'
        elif attacker == author1:
            group = 'attacker_initiated'
        elif attacker == author2:
            group = 'non_attacker_initiated'
        else:
            group = 'attacker_not_in_opening_exchange'
        cohort.at[cid, 'role_group'] = group
    return first, second, cohort


def compute_opening_results(first, second, cohort, args):
    # Import the dependency-light statistical helper without loading matplotlib.
    from error_analysis import bh_adjust
    requested = list(dict.fromkeys(args.opening_features + args.opening_binary_features))
    overlapping = set(args.opening_features) & set(args.opening_binary_features)
    if overlapping:
        raise ValueError(f'Features cannot be both categorical and binary: {sorted(overlapping)}')
    features = [name for name in requested if name in first]
    if not features:
        raise ValueError(f'None of the requested opening features were found: {requested}')
    rows, coverage = [], []
    for feature in features:
        binary = feature in args.opening_binary_features
        x1 = first[feature].map(lambda v: feature_set(v, binary, args.multilabel_separator))
        x2 = second[feature].map(lambda v: feature_set(v, binary, args.multilabel_separator))
        complete = x1.notna() & x2.notna()
        local = cohort.loc[complete]
        categories = sorted(frozenset().union(*x1.loc[complete], *x2.loc[complete]))
        if binary:
            categories = ['present']
        coverage.append(dict(feature=feature, complete_conversations=int(complete.sum()),
                             missing_feature_conversations=int((~complete).sum())))
        for panel, subgroup in [('A', None), ('B', 'attacker_initiated'), ('C', 'non_attacker_initiated')]:
            positive = local.index[(local.label == 1) & ((local.role_group == subgroup) if subgroup else True)]
            negative = local.index[local.label == 0]
            for position, values in [(1, x1), (2, x2)]:
                role = 'all' if panel == 'A' else 'attacker' if (panel == 'B' and position == 1) or (panel == 'C' and position == 2) else 'non_attacker'
                for category in categories:
                    a = sum(category in values.loc[cid] for cid in positive)
                    c = sum(category in values.loc[cid] for cid in negative)
                    stats = effect(a, len(positive), c, len(negative), args.opening_pseudocount)
                    supported = min(len(positive), len(negative)) >= args.min_support and a+c >= args.opening_min_count
                    variable = a+c > 0 and a+c < len(positive)+len(negative)
                    valid = supported and variable
                    rows.append(dict(panel=panel, position=position, role=role, feature=feature, category=category,
                                     n_awry=len(positive), present_awry=a, n_on_track=len(negative), present_on_track=c,
                                     **stats, p_value=stats[args.opening_test+'_p'] if valid else np.nan,
                                     test=args.opening_test, supported=valid,
                                     status='ok' if valid else 'insufficient_support_or_constant'))
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError('No labeled opening features available after excluding missing annotations')
    table['q_value_bh'] = bh_adjust(table.p_value)
    chosen = table.p_value if args.opening_significance == 'raw' else table.q_value_bh
    table['significant'] = chosen < .05
    table['solid'] = table.significant & (table.log_odds.abs() >= args.opening_effect_threshold)
    table['significance_marks'] = [significance_marks(p, '*' if pos == 1 else '+') for p,pos in zip(chosen, table.position)]
    return table, coverage, [name for name in requested if name not in features]


def plot_openings(table, output, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    drawable = table.loc[table.supported & np.isfinite(table.log_odds)]
    ordered = drawable.assign(magnitude=drawable.log_odds.abs()).groupby(['feature','category']).magnitude.max().sort_values(ascending=False)
    keys = list(ordered.head(args.opening_max_features).index)
    fig, axes = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(16, max(5, .42*len(keys)+2.5)))
    titles = ['A. First and second comments', 'B. Attacker initiated', 'C. Non-attacker initiated']
    for panel, ax, title in zip('ABC', axes, titles):
        ax.axvline(0, color='#999999', lw=1)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Natural log-odds ratio\n← on-track     awry →')
        for spine in ['top','right']:
            ax.spines[spine].set_visible(False)
        panel_rows = table.loc[(table.panel == panel) & table.supported]
        for y, (feature, category) in enumerate(keys):
            for row in panel_rows.loc[(panel_rows.feature == feature) & (panel_rows.category == category)].itertuples():
                color = PURPLE if row.position == 1 else GREEN
                marker = ('D' if row.position == 1 else 's') if panel == 'A' else ('v' if row.role == 'attacker' else 'o')
                offset = -.13 if row.position == 1 else .13
                ax.errorbar(row.log_odds, y+offset, xerr=[[row.log_odds-row.ci_low],[row.ci_high-row.log_odds]],
                            fmt='none', ecolor=color, alpha=.35, capsize=2, lw=.8)
                ax.scatter(row.log_odds, y+offset, marker=marker, s=40, edgecolors=color,
                           facecolors=color if row.solid else 'none', zorder=3)
                if panel == 'A' and row.significance_marks:
                    ax.annotate(row.significance_marks, (row.log_odds, y+offset), xytext=(6,3), textcoords='offset points', color=color, fontsize=8)
        if panel_rows.empty:
            ax.text(.5, .5, 'No eligible comparisons\n(check metadata and support)', transform=ax.transAxes,
                    ha='center', va='center', color='#666666')
        ax.grid(axis='y', alpha=.12)
    axes[0].set_yticks(range(len(keys)))
    axes[0].set_yticklabels([feature if category == 'present' else f'{feature}: {category}' for feature,category in keys], fontsize=9)
    axes[0].invert_yaxis()
    if not keys:
        axes[0].text(.5,.2,'No feature meets the support filters', transform=axes[0].transAxes, ha='center')
    legend = [Line2D([],[],color=PURPLE,marker='D',ls='',label='First comment (purple)'),
              Line2D([],[],color=GREEN,marker='s',ls='',label='Second comment (green)'),
              Line2D([],[],color='gray',marker='v',ls='',label='Eventual attacker (B/C)'),
              Line2D([],[],color='gray',marker='o',ls='',label='Non-attacker (B/C)')]
    fig.legend(handles=legend, loc='lower center', ncol=2, bbox_to_anchor=(.5,.01), fontsize=9)
    significance = 'p' if args.opening_significance == 'raw' else 'BH q'
    fig.suptitle(f'Opening features vs. actual conversation outcome\nFilled: {significance} < 0.05 and |log-odds| ≥ {args.opening_effect_threshold:g}; bars: approximate 95% intervals', fontsize=12)
    fig.tight_layout(rect=[0,.09,1,.94])
    for ext in ('png','svg','pdf'):
        fig.savefig(output/f'opening_log_odds.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    return len(keys)


def analyze_openings(joined, args):
    if args.opening_pseudocount <= 0 or not np.isfinite(args.opening_pseudocount):
        raise ValueError('--opening_pseudocount must be finite and positive')
    if args.opening_effect_threshold < 0 or not np.isfinite(args.opening_effect_threshold):
        raise ValueError('--opening_effect_threshold must be finite and nonnegative')
    if args.opening_min_count < 1 or args.opening_max_features < 1:
        raise ValueError('Opening plot counts must be positive')
    first, second, cohort = opening_cohort(joined, args)
    table, coverage, missing = compute_opening_results(first, second, cohort, args)
    output = Path(args.output_dir)/'opening_analysis'
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output/'log_odds.csv', index=False)
    cohort.to_csv(output/'cohort.csv')
    count = plot_openings(table, output, args)
    audit = dict(input_conversations=joined.conversation_id.nunique(), eligible_conversations=len(cohort),
                 excluded_missing_opening_comments=joined.conversation_id.nunique()-len(cohort),
                 role_groups={str(k):int(v) for k,v in cohort.role_group.value_counts().items()},
                 feature_coverage=coverage, absent_features=missing, plotted_features=count,
                 comparisons=len(table), valid_tests=int(table.p_value.notna().sum()), arguments=vars(args))
    (output/'audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False)+'\n')
    report = f'''# Opening-exchange log-odds analysis

Inspired by [Figure 2 of Conversations Gone Awry (Zhang et al., 2018)](https://aclanthology.org/P18-1125/).
This is an explicit implementation on your annotations, not a claim of identical feature extraction or exact paper replication.

![Opening log-odds](opening_log_odds.png)

## Cohort and panels

- Actual outcome `label`: 1 = awry, 0 = on-track. These are not model errors or predictions.
- Only comments at timestep 1 and 2 are features. Both must be observed and annotated; {len(cohort)} conversations retained.
- Panel A: all eligible awry vs. on-track conversations, separately at each position.
- Panel B: awry conversations where the eventual attacker wrote comment 1; panel C: where the attacker wrote comment 2. Each is compared with **all eligible on-track conversations at the same comment position**, not with fictitious attackers in on-track conversations.
- Explicit attacker identity and two distinct, nonempty author IDs are required for B/C. Unknown identities, a later third-party attacker, or the same opening author are excluded from B/C, retained in A. No attacker is inferred from `aggressive`.
- For each feature, both opening annotations must be nonmissing. An explicit empty JSON list (categorical) or false/0 (binary) is a known absence. Blank is missing. Counts and exclusions are in audit.json.
- Role groups: {audit['role_groups']}. Absent requested feature columns: {missing}.

## Statistics and symbols

For feature counts a/b (present/absent in awry) and c/d (on-track), the effect is
`ln(((a+s)/(b+s))/((c+s)/(d+s)))`, with s={args.opening_pseudocount} added to all four cells.
Positive values favor awry. Intervals are approximate Wald intervals with standard error
`sqrt(1/(a+s)+1/(b+s)+1/(c+s)+1/(d+s))`; they are not exact test inversions.

Both p-values are exported on **unsmoothed** counts:
- `binomial_p`: two-sided one-sample binomial test of a successes in n_awry trials against p0=c/n_on_track. This plug-in null treats estimated on-track prevalence as fixed and ignores its sampling uncertainty. Boundary p0=0 or 1 can yield extreme p-values. It is **not** a paired binomial test or an exact two-sample test.
- `fisher_p`: two-sided Fisher exact test of the 2×2 table; use `--opening_test fisher` for a two-sample comparison.

Selected test: **{args.opening_test}**. Marker significance: **{args.opening_significance}**.
BH q-values cover all supported selected-test comparisons across all features, positions and panels **before plot filtering**.
Raw p-values are the default to follow the caption; inspect q-values to assess multiple comparisons.
At least {args.min_support} conversations per outcome group and {args.opening_min_count} feature occurrences are required; constant features are not tested.

Purple = first comment, green = second. A uses diamonds/squares; B/C use downward triangles for attackers and circles for non-attackers.
Filled markers require significance < .05 and absolute log-odds ≥ {args.opening_effect_threshold}.
In A, * / ** / *** mean < .05 / .01 / .001 for comment 1; + / ++ / +++ mean the same for comment 2.
The caption's “!” and “****” are corrected to square and **, respectively, as in the paper.
Up to {args.opening_max_features} rows are plotted, ordered by maximum supported absolute effect; every comparison remains in log_odds.csv.

## Limits

These are unadjusted observational associations, not causal effects. The tests assume independent conversations;
shared threads/authors or matched conversation pairs require an appropriate clustered/paired analysis.
This implementation does not reconstruct the paper's matching or learn its politeness/prompt detectors.
`annot1` and `annot2` remain separate code sets; speech-act `inference_type` is not automatically renamed to prompt type.
Attacker identity is retrospective metadata, not a deployable forecasting feature.
Repeat separately by seed if integrating with prediction outputs; the opening results should be unchanged when cohorts and true labels are identical.
'''
    (output/'report.md').write_text(report)
    print(f'Opening analysis: {output / "report.md"}')
    return audit
