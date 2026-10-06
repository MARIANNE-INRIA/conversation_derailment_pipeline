#!/usr/bin/env python3
"""Associate forecast errors with message annotations, without fitting a test threshold."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, pointbiserialr

CATEGORICAL = ['inference_type', 'as_intended', 'PRE/IMP', 'aggressive', 'annot1', 'annot2']
TEXT = ['Pragmatic_Inferences', 'most_salient_inference']
MISSING = '__MISSING__'


def read_csv(path, sep=','):
    return pd.read_csv(path, dtype=str, keep_default_na=False, sep=sep)


def tokens(value, separator=r'[+;|]'):
    value = str(value).strip()
    if not value:
        return frozenset([MISSING])
    # Structured lists are supported; otherwise separators are explicit and configurable.
    if value.startswith('['):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list) and all(isinstance(x, (str, int)) for x in parsed):
                return frozenset(str(x).strip().casefold() for x in parsed if str(x).strip()) or frozenset([MISSING])
        except json.JSONDecodeError:
            pass
    return frozenset(part.strip().casefold() for part in re.split(separator, value) if part.strip()) or frozenset([MISSING])


def boolean_tokens(value):
    value = str(value).strip().casefold()
    if not value:
        return frozenset([MISSING])
    if value in ('true', 'yes', '1', 'oui'):
        return frozenset(['true'])
    if value in ('false', 'no', '0', 'non'):
        return frozenset(['false'])
    return frozenset(['unknown:' + value])


def parse_inferences(value):
    text = str(value).strip()
    if not text:
        return MISSING, np.nan
    if text.casefold() in ('literal', 'none', '[]', '{}'):
        return 'none_or_literal', 0.0
    # Some human exports contain comma-separated JSON dictionaries without brackets.
    for candidate in (text, '[' + text.rstrip(',') + ']'):
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, list):
            return 'structured', float(len(parsed))
        if isinstance(parsed, dict):
            return 'structured', float(1 if 'content' in parsed else len(parsed))
    return 'free_text_or_unparsed', np.nan


def normalized_keys(frame, keys):
    frame = frame.copy()
    for key in keys:
        if key not in frame:
            raise ValueError(f'Missing join column: {key}')
        frame[key] = frame[key].astype(str).str.strip()
        if frame[key].eq('').any():
            raise ValueError(f'Empty join key: {key}')
        if key == 'timestep':
            values = pd.to_numeric(frame[key], errors='raise')
            if (values < 1).any() or (values % 1 != 0).any():
                raise ValueError('Timesteps must be positive integers, starting at 1')
            frame[key] = values.astype('int64')
    return frame


def load_annotations(path, conv_col, turn_col, filename_ids, sep, extra_columns=()):
    path = Path(path)
    paths = sorted(path.glob('*.csv')) if path.is_dir() else [path]
    if not paths:
        raise ValueError('No annotation CSV files found')
    frames = []
    for file in paths:
        frame = read_csv(file, sep)
        if 'message_info' in frame:
            # Exports contain prefix-local metadata; take the last observed message.
            rows = []
            for row in frame.to_dict('records'):
                info = json.loads(row['message_info'])
                t = int(row['timestep'])
                if not isinstance(info, list) or len(info) != t:
                    raise ValueError('message_info must contain exactly timestep observed messages')
                rows.append({**info[-1], 'conversation_id': row['conversation_id'], 'timestep': t})
            frame = pd.DataFrame(rows)
        else:
            if filename_ids:
                frame['conversation_id'] = re.sub(r'_(NOT_TOXIC|TOXIC)$', '', file.stem)
            elif conv_col != 'conversation_id':
                frame = frame.rename(columns={conv_col: 'conversation_id'})
            if turn_col != 'timestep':
                frame = frame.rename(columns={turn_col: 'timestep'})
        frame = normalized_keys(frame, ['conversation_id', 'timestep'])
        available = [c for c in dict.fromkeys(CATEGORICAL + TEXT + list(extra_columns)) if c in frame and c not in ('conversation_id', 'timestep')]
        frames.append(frame[['conversation_id', 'timestep'] + available])
    combined = pd.concat(frames, ignore_index=True).fillna('')
    # Repeated exports may repeat the same annotation, never average conflicting raters.
    combined = combined.drop_duplicates()
    if combined.duplicated(['conversation_id', 'timestep']).any():
        raise ValueError('Conflicting annotations for a conversation/timestep; select one annotation source or adjudicate first')
    return combined


def prepare_predictions(frame, threshold=None, require_prediction=True):
    frame = normalized_keys(frame, ['conversation_id', 'timestep'])
    if frame.duplicated(['conversation_id', 'timestep']).any():
        raise ValueError('Duplicate predictions: analyze each model/seed separately')
    frame['label'] = pd.to_numeric(frame['label'], errors='raise')
    if not frame['label'].isin([0, 1]).all():
        raise ValueError('label must contain only 0/1')
    if (frame.groupby('conversation_id')['label'].nunique() != 1).any():
        raise ValueError('Conflicting ground-truth labels within a conversation')
    if threshold is not None:
        if not 0 <= threshold <= 1:
            raise ValueError('threshold must be in [0, 1]')
        frame['probability'] = pd.to_numeric(frame['probability'], errors='raise')
        if not frame['probability'].between(0, 1).all():
            raise ValueError('Invalid or missing probabilities')
        frame['prediction'] = (frame['probability'] > threshold).astype(int)
    elif 'prediction' in frame:
        frame['prediction'] = pd.to_numeric(frame['prediction'], errors='raise')
        if not frame['prediction'].isin([0, 1]).all():
            raise ValueError('prediction must contain only 0/1')
    elif require_prediction:
        raise ValueError('Provide --threshold_json/--threshold (selected on validation), or a prediction column')
    if 'total_turns' in frame:
        total = pd.to_numeric(frame['total_turns'], errors='raise')
        if (total < 1).any() or (total % 1 != 0).any():
            raise ValueError('total_turns must contain positive integers')
        frame['total_turns'] = total.astype('int64')
        if (frame['timestep'] > total - frame['label']).any():
            raise ValueError('Predictions include a future/terminal attack turn; expected forecasting prefixes')
    # Conversation-level OR aggregation needs every observed prefix, including earlier alerts.
    for _, group in frame.groupby('conversation_id'):
        steps = sorted(group['timestep'])
        if 'total_turns' in group and (group.total_turns.nunique() != 1 or max(steps) != int(group.total_turns.iloc[0] - group.label.iloc[0])):
            raise ValueError('Missing final observed prefixes or conflicting total_turns in a conversation')
        if steps != list(range(1, max(steps) + 1)):
            raise ValueError('Prefix predictions must contain contiguous timesteps starting at 1')
    return frame


def confusion(frame):
    frame = frame.copy()
    frame['error'] = (frame['prediction'] != frame['label']).astype(int)
    frame['error_type'] = np.select([
        (frame.label == 1) & (frame.prediction == 1), (frame.label == 0) & (frame.prediction == 1),
        (frame.label == 1) & (frame.prediction == 0)], ['TP', 'FP', 'FN'], default='TN')
    return frame


def make_features(joined, separator):
    categorical, numeric = {}, {}
    for name in CATEGORICAL:
        if name in joined:
            categorical[name] = joined[name].map(boolean_tokens if name == 'as_intended' else lambda v: tokens(v, separator))
    if 'annot1' in categorical and 'annot2' in categorical:
        def agreement(pair):
            a, b = pair
            return frozenset([MISSING if MISSING in a or MISSING in b else 'agree' if a == b else 'disagree'])
        categorical['annotator_agreement'] = pd.Series(
            [agreement(pair) for pair in zip(categorical['annot1'], categorical['annot2'])], index=joined.index)
    if 'Pragmatic_Inferences' in joined:
        parsed = joined['Pragmatic_Inferences'].map(parse_inferences)
        categorical['Pragmatic_Inferences.format'] = parsed.map(lambda x: frozenset([x[0]]))
        numeric['Pragmatic_Inferences.count'] = parsed.map(lambda x: x[1])
    if 'most_salient_inference' in joined:
        categorical['most_salient_inference.presence'] = joined['most_salient_inference'].map(
            lambda x: frozenset(['present' if str(x).strip() else MISSING]))
        numeric['most_salient_inference.words'] = joined['most_salient_inference'].map(
            lambda x: float(len(str(x).split())) if str(x).strip() else np.nan)
    return categorical, numeric


def build_units(joined, level, separator):
    categories, numbers = make_features(joined, separator)
    if level == 'prefix':
        return confusion(joined[['conversation_id', 'timestep', 'label', 'prediction']]), categories, numbers
    units = joined.groupby('conversation_id', sort=True).agg(label=('label', 'first'), prediction=('prediction', 'max'))
    categories = {key: values.groupby(joined.conversation_id).agg(lambda cells: frozenset().union(*cells))
                  for key, values in categories.items()}
    numbers = {key: values.groupby(joined.conversation_id).mean() for key, values in numbers.items()}
    return confusion(units), categories, numbers


def bh_adjust(values):
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    order = valid[np.argsort(values[valid])]
    if len(order):
        adjusted = values[order] * len(order) / np.arange(1, len(order) + 1)
        result[order] = np.minimum(1, np.minimum.accumulate(adjusted[::-1])[::-1])
    return result


def associations(units, categories, numbers, min_support, inferential):
    categorical_rows, numeric_rows = [], []
    for target, subset in [('error', units.index), ('false_positive', units.index[units.label == 0]),
                            ('false_negative', units.index[units.label == 1])]:
        y = units.loc[subset, 'error'].to_numpy(dtype=int)
        for feature, values in categories.items():
            values = values.loc[subset]
            labels = sorted(frozenset().union(*values)) if len(values) else []
            for label in labels:
                x = values.map(lambda v: label in v).to_numpy(dtype=bool)
                a, b = int((x & (y == 1)).sum()), int((x & (y == 0)).sum())
                c, d = int((~x & (y == 1)).sum()), int((~x & (y == 0)).sum())
                den = float((a+b)*(c+d)*(a+c)*(b+d)) ** .5
                rate_with = a/(a+b) if a+b else np.nan
                rate_without = c/(c+d) if c+d else np.nan
                status = 'ok' if min(a+b, c+d) >= min_support and den else 'insufficient_support_or_constant'
                p = float(fisher_exact([[a,b],[c,d]])[1]) if inferential and status == 'ok' else np.nan
                categorical_rows.append(dict(feature=feature, category=label, target=target,
                    n_with=a+b, errors_with=a, n_without=c+d, errors_without=c,
                    rate_with=rate_with, rate_without=rate_without, risk_difference=rate_with-rate_without,
                    phi=(a*d-b*c)/den if den else np.nan,
                    odds_ratio=(a*d)/(b*c) if b*c else np.inf if a*d else np.nan,
                    p_value=p, status=status if inferential else 'descriptive_prefix_dependence'))
        for feature, values in numbers.items():
            x = values.loc[subset].to_numpy(dtype=float)
            valid = np.isfinite(x)
            x, outcome = x[valid], y[valid]
            valid_test = len(x) >= min_support and len(set(x)) > 1 and len(set(outcome)) > 1
            coefficient, p = pointbiserialr(outcome, x) if valid_test else (np.nan, np.nan)
            numeric_rows.append(dict(feature=feature, target=target, n=len(x),
                mean_error=float(x[outcome == 1].mean()) if (outcome == 1).any() else np.nan,
                mean_correct=float(x[outcome == 0].mean()) if (outcome == 0).any() else np.nan,
                point_biserial_r=float(coefficient), p_value=float(p) if inferential else np.nan))
    # One family for all features, levels, and error targets in this run.
    rows = categorical_rows + numeric_rows
    for row, q in zip(rows, bh_adjust([row['p_value'] for row in rows])):
        row['q_value_bh'] = q
    return pd.DataFrame(categorical_rows), pd.DataFrame(numeric_rows, columns=[
        'feature', 'target', 'n', 'mean_error', 'mean_correct', 'point_biserial_r', 'p_value', 'q_value_bh'])


def run(args):
    threshold = args.threshold
    if args.threshold_json:
        threshold = float(json.loads(Path(args.threshold_json).read_text())['best_tau'])
    opening_only = getattr(args, 'opening_only', False)
    opening_enabled = opening_only or getattr(args, 'opening_analysis', False)
    predictions = prepare_predictions(read_csv(args.predictions, args.sep), threshold, require_prediction=not opening_only)
    extras = []
    if opening_enabled:
        reserved = {'conversation_id', 'timestep', 'label', 'prediction', 'probability', 'total_turns', '_merge'}
        requested = args.opening_features + args.opening_binary_features
        if set(requested) & (reserved | {args.opening_author_col, args.opening_attacker_col}):
            raise ValueError('Opening features must be annotations, not outcomes, model outputs, keys or author/attacker IDs')
        if {args.opening_author_col, args.opening_attacker_col} & reserved:
            raise ValueError('Author/attacker columns cannot be outcome, prediction or join-key columns')
        extras = args.opening_features + args.opening_binary_features + [args.opening_author_col, args.opening_attacker_col]
    annotations = load_annotations(args.annotations, args.annotation_conversation_col,
                                   args.annotation_turn_col, args.filename_ids, args.sep, extra_columns=extras)
    available = [c for c in CATEGORICAL + TEXT if c in annotations]
    if not available and not opening_only:
        raise ValueError('None of the supported annotation columns were found')
    # Use annotations as the authoritative source, even if predictions include namesakes.
    predictions = predictions.drop(columns=[c for c in set(CATEGORICAL + TEXT + extras) if c in predictions])
    joined = predictions.merge(annotations, on=['conversation_id','timestep'], how='left', validate='one_to_one', indicator=True)
    if opening_only:
        from opening_analysis import analyze_openings
        return analyze_openings(joined, args)
    unmatched = joined['_merge'] != 'both'
    excluded_conversations = set(joined.loc[unmatched, 'conversation_id'])
    if unmatched.any() and not args.allow_partial:
        examples = joined.loc[unmatched, ['conversation_id', 'timestep']].head(3).to_dict('records')
        raise ValueError(f'{int(unmatched.sum())} prefixes have no annotation row; examples: {examples}. '
                         'Check join keys, or explicitly use --allow_partial.')
    selected = joined.loc[~joined.conversation_id.isin(excluded_conversations)] if args.level == 'conversation' else joined.loc[~unmatched]
    selected = selected.drop(columns='_merge').fillna('').reset_index(drop=True)
    if selected.empty:
        raise ValueError('No fully matched analysis units remain')
    units, categories, numbers = build_units(selected, args.level, args.multilabel_separator)
    categorical, numeric = associations(units, categories, numbers, args.min_support, args.level == 'conversation')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    qualitative = confusion(selected)
    if args.level == 'conversation':
        qualitative['conversation_error_type'] = qualitative.conversation_id.map(units.error_type)
        errors = qualitative.loc[qualitative.conversation_error_type.isin(['FP','FN'])]
    else:
        errors = qualitative.loc[qualitative.error == 1]
    qualitative.to_csv(output/'annotated_predictions.csv', index=False)
    errors.to_csv(output/'errors.csv', index=False)
    exported = units.copy()
    for key, values in categories.items():
        exported[key] = values.map(lambda v: json.dumps(sorted(v), ensure_ascii=False))
    for key, values in numbers.items():
        exported[key] = values
    exported.to_csv(output/'analysis_units.csv', index=args.level == 'conversation')
    categorical.to_csv(output/'categorical_associations.csv', index=False)
    numeric.to_csv(output/'numeric_associations.csv', index=False)
    joined.loc[unmatched, ['conversation_id','timestep']].to_csv(output/'unmatched_predictions.csv', index=False)
    audit = dict(level=args.level, threshold=threshold, threshold_rule='probability > threshold' if threshold is not None else 'existing prediction column',
                 prediction_rows=len(predictions), annotation_rows=len(annotations), unmatched_prediction_rows=int(unmatched.sum()),
                 excluded_conversations=len(excluded_conversations) if args.level == 'conversation' else 0,
                 analyzed_units=len(units), confusion={key:int((units.error_type == key).sum()) for key in ['TP','TN','FP','FN']},
                 available_columns=available, absent_columns=[c for c in CATEGORICAL+TEXT if c not in available],
                 blank_annotation_cells={c:int(selected[c].eq('').sum()) for c in available},
                 min_support=args.min_support, arguments=vars(args))
    (output/'audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False)+'\n')
    top = categorical.sort_values(['q_value_bh','risk_difference'], ascending=[True,False], na_position='last').head(20)
    lines = ['# Analyse des erreurs', '', f'Unités analysées : {len(units)} ({args.level}).',
             f'Confusion : {audit["confusion"]}.', f'Préfixes non appariés : {int(unmatched.sum())}.', '',
             '## Interprétation', '',
             '- Une catégorie est présente si elle apparaît dans au moins un message observé de la conversation. En mode prefix, seule l’annotation du dernier message observé est utilisée.',
             '- FP : prédiction positive parmi les conversations négatives ; FN : prédiction négative parmi les positives. Le label cible reste le label de dérive de la conversation, pas un label d’agressivité du message.',
             '- Phi positif / odds ratio > 1 : davantage d’erreurs en présence du label. Les catégories se chevauchent ; les effets ne sont pas ajustés pour les autres labels, la longueur ou le corpus.',
             '- Les tests de Fisher et point-bisériaux supposent des conversations indépendantes. Les branches partageant des messages violent cette hypothèse : les p-values sont alors exploratoires. Aucune p-value n’est calculée en mode prefix.',
             '- Correction Benjamini–Hochberg sur tous les tests valides de ce lancement. Ne pas sélectionner un modèle ou un seuil avec ces résultats sur le test.',
             '- Les textes libres restent dans errors.csv. Les variables dérivées (format, nombre d’inférences structurées, présence et longueur du texte saillant) ne mesurent pas leur sens. Les textes non parsables n’ont pas de nombre d’inférences inventé.',
             '- annot1 et annot2 sont analysés séparément. Leur accord est l’égalité des ensembles de codes normalisés, sans consensus automatique. Les valeurs originales sont conservées.',
             '- Les annotations sans prédiction correspondante ne sont pas utilisées, notamment le dernier message toxique exclu des préfixes. aggressive peut être lié à la construction de la cible : une association ne prouve pas une cause.',
             '', '## Associations (premières lignes)', '', '```csv', top.to_csv(index=False).strip(), '```', '',
             'Méthodes : [Fisher exact](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html), '
             '[correction FDR](https://www.statsmodels.org/v0.11.1/generated/statsmodels.stats.multitest.fdrcorrection.html).', '']
    (output/'report.md').write_text('\n'.join(lines))
    print(json.dumps({k:audit[k] for k in ['analyzed_units','confusion','unmatched_prediction_rows','absent_columns']}, indent=2))
    print(f'Report: {output / "report.md"}')
    if opening_enabled:
        from opening_analysis import analyze_openings
        analyze_openings(joined, args)
        with (output / 'report.md').open('a') as handle:
            handle.write('\n## Opening exchange analysis\n\nSee [opening_analysis/report.md](opening_analysis/report.md) for outcome-based log-odds and panels A/B/C.\n')
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True, help='Prefix-level predictions for one model and one seed')
    parser.add_argument('--annotations', required=True, help='Annotated CSV, directory of CSVs, or test CSV containing message_info')
    parser.add_argument('--output_dir', required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--threshold_json', help='Validation best_tau.json; never optimize a threshold here')
    group.add_argument('--threshold', type=float)
    parser.add_argument('--level', choices=['conversation','prefix'], default='conversation')
    parser.add_argument('--annotation_conversation_col', default='conversation_id')
    parser.add_argument('--annotation_turn_col', default='timestep', help='Use Turn_ID or turn_index if appropriate')
    parser.add_argument('--filename_ids', action='store_true', help='Explicitly derive IDs from filenames, stripping _TOXIC/_NOT_TOXIC')
    parser.add_argument('--multilabel_separator', default=r'[+;|]', help='Regex; default splits +, ;, | but preserves slash in PRE/IMP')
    parser.add_argument('--sep', default=',')
    parser.add_argument('--min_support', type=int, default=5)
    parser.add_argument('--allow_partial', action='store_true', help='Exclude incomplete conversations (or unmatched rows in prefix mode)')
    from opening_analysis import add_opening_arguments
    add_opening_arguments(parser)
    args = parser.parse_args()
    if args.min_support < 1:
        parser.error('--min_support must be positive')
    try:
        run(args)
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
