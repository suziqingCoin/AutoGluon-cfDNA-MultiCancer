"""Shared I/O and scoped manuscript styling for authorized supplementary replay.

No model loading, inference, fitting, resplitting or test-based model selection is performed.
Predictor-shaped adapters only retrieve frozen, patient-aligned predictions.
Public aggregate redraws remain independently implemented in plot_public.py.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.font_manager import FontProperties, findfont

MODELS = ['XGBoost', 'RandomForest', 'LightGBM', 'NeuralNetFastAI', 'WeightedEnsemble_L2']


@dataclass
class ReplayContext:
    """Explicit task/method inputs; patient files are read lazily when required."""
    task_dir: Path
    method: str
    output_dir: Path
    _prediction_cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.task_dir = Path(self.task_dir).resolve()
        self.output_dir = Path(self.output_dir).resolve()

    @property
    def figures(self):
        return self.task_dir / 'figures'

    @property
    def features(self):
        return self.figures / 'data' / self.method / 'selected_feature_scores.csv'

    @property
    def curve(self):
        return self.figures / 'data' / self.method / 'K_optimization.csv'

    @cached_property
    def split(self):
        path = self.task_dir / 'results/holdout/patient_split.csv'
        if not path.is_file():
            raise FileNotFoundError('Authorized patient_split.csv required; see DATA_ACCESS.md')
        data = pd.read_csv(path).rename(columns={'__target__': 'class'}).set_index('SampleID', drop=False)
        return data[data.part.ne('evaluation')].copy(), data[data.part.eq('evaluation')].copy()

    @property
    def train_data(self):
        return self.split[0]

    @property
    def test_data(self):
        return self.split[1]

    @cached_property
    def data(self):
        return pd.concat(self.split)

    def locked_split(self, data=None, *args, **kwargs):
        """Return the recorded split; historical split arguments do not resplit patients."""
        return self.split

    def predictions(self, data, model):
        model = model.replace('_BAG_L1', '')
        if model not in self._prediction_cache:
            base = self.task_dir / 'results/holdout/none' / self.method
            train = pd.read_csv(self.task_dir / 'results/recovered_training_predictions' / self.method /
                                f'apparent_training_predictions_{model}.csv')
            test = pd.read_csv(base / f'evaluation_predictions_{model}.csv')
            both = pd.concat([train, test]).set_index('SampleID', drop=False)
            assert both.index.is_unique
            self._prediction_cache[model] = both
        return self._prediction_cache[model].loc[data.SampleID]

    @cached_property
    def predictor(self):
        return FrozenPredictor(self)

    @cached_property
    def predictor_api(self):
        return _PredictorAPI(self)


class _FrozenModel:
    def __init__(self, context, name):
        self.context = context
        self.name = name

    def predict_proba(self, data):
        values = self.context.predictions(data, self.name)
        return values.p_1.to_numpy() if 'p_2' not in values else values[[f'p_{i}' for i in range(6)]].to_numpy()


class FrozenPredictor:
    """Use saved probabilities and recover each method's choice from frozen validation metrics."""
    def __init__(self, context):
        self.context = context
        self._trainer = self
        self.label = 'class'
        self.class_labels = list(range(6 if context.task_dir.name == 'pancancer' else 2))
        values = pd.read_csv(context.task_dir / 'results/holdout/all_none_inner_validation_metrics.csv')
        values = values[values.method.eq(context.method)]
        self.model_best = sorted(values.to_dict('records'), key=lambda row:
            tuple(-round(row[k], 12) for k in ['AUC', 'AUPRC', 'Recall', 'Specificity']) +
            (row['k'], row['model']))[0]['model']

    def load_model(self, name):
        return _FrozenModel(self.context, name)

    def model_names(self):
        return list(MODELS)

    def predict_proba(self, data, model=None, **kwargs):
        values = self.context.predictions(data, model or self.model_best)
        return pd.DataFrame(values[[f'p_{i}' for i in self.class_labels]].to_numpy(),
                            index=data.index, columns=self.class_labels)

    def predict(self, data, model=None):
        values = self.context.predictions(data, model or self.model_best)
        return pd.Series(values.y_pred.to_numpy(), index=data.index)


class _PredictorAPI:
    def __init__(self, context):
        self.context = context

    def load(self, *args, **kwargs):
        return self.context.predictor


@contextmanager
def manuscript_style(context, names, *, save=True, capture=None):
    """Apply the prior renderer's Arial/regular style only for this replay.

    Save both vector PDF and 300-dpi PNG with the original crop/padding. All global
    hooks, rcParams and RNG state are restored on exit; imports have no font hooks.
    """
    findfont(FontProperties(family='Arial'), fallback_to_default=False)
    original = dict(weight=Text.set_fontweight, alias_weight=Text.set_weight,
                    family=Text.set_fontfamily, props=Text.set_fontproperties,
                    save=Figure.savefig, show=plt.show)
    rng_state = np.random.get_state()
    count = 0
    outputs = []
    sizes = []
    cmap = matplotlib.cm
    added_get_cmap = not hasattr(cmap, 'get_cmap')
    if added_get_cmap:
        cmap.get_cmap = lambda name=None, lut=None: matplotlib.colormaps.get_cmap(name)

    def fixed_props(self, value):
        props = FontProperties._from_any(value).copy()
        props.set_family('Arial')
        props.set_weight('normal')
        return original['props'](self, props)

    def save_figure(fig, filename, **kwargs):
        nonlocal count
        if count >= len(names):
            raise AssertionError('Plot saved more figures than its manifest declares')
        name = names[count]
        count += 1
        for text in fig.findobj(Text):
            text.set_fontfamily('Arial')
            text.set_fontweight('normal')
        for artist in fig.findobj():
            if hasattr(artist, 'set_rasterized') and artist.get_rasterized():
                artist.set_rasterized(False)
        if capture is not None:
            capture(fig, name)
        if save:
            context.output_dir.mkdir(parents=True, exist_ok=True)
            for ext in ['pdf', 'png']:
                target = context.output_dir / f'{name}.{ext}'
                original['save'](fig, target, format=ext, bbox_inches='tight', pad_inches=.08, dpi=300)
                outputs.append(target.name)
        sizes.append(fig.get_size_inches().tolist())

    try:
        with plt.rc_context():
            Text.set_fontweight = lambda self, value: original['weight'](self, 'normal')
            Text.set_weight = Text.set_fontweight
            Text.set_fontfamily = lambda self, value: original['family'](self, 'Arial')
            Text.set_fontproperties = fixed_props
            plt.rcdefaults()
            plt.rcParams.update({'font.family': 'Arial', 'font.weight': 'normal',
                'axes.titleweight': 'normal', 'axes.labelweight': 'normal',
                'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.unicode_minus': False})
            Figure.savefig = save_figure
            plt.show = lambda *args, **kwargs: plt.close('all')
            np.random.seed(42)
            yield {'outputs': outputs, 'canvas_inches': sizes}
            assert count == len(names), (count, names)
    finally:
        Text.set_fontweight = original['weight']
        Text.set_weight = original['alias_weight']
        Text.set_fontfamily = original['family']
        Text.set_fontproperties = original['props']
        Figure.savefig = original['save']
        plt.show = original['show']
        if added_get_cmap:
            del cmap.get_cmap
        np.random.set_state(rng_state)
        plt.close('all')


def main(module_path, functions, argv=None):
    """Run semantic plotting functions declared in this task's source manifest."""
    module_path = Path(module_path).resolve()
    task_dir = module_path.parents[2]
    entries = json.loads((task_dir / 'figures/plot_manifest.json').read_text())
    names = [name for entry in entries for name in entry['names']]
    parser = argparse.ArgumentParser(description=functions.get('__doc__', __doc__))
    parser.add_argument('--list', action='store_true', help='List plots and input requirements without reading private files')
    parser.add_argument('--only', nargs='+', choices=names, help='Replay selected plotting groups (a model ROC group emits all five model plots)')
    parser.add_argument('--output-dir', type=Path, default=task_dir.parent / 'private/replay' / task_dir.name)
    args = parser.parse_args(argv)
    if args.list:
        print(json.dumps([dict(figure=e['supplementary_figure'], names=e['names'],
            function=e['function'], kwargs=e.get('kwargs', {}),
            requires_private_data=e['requires_private_data'], public_data=e['public_data'],
            private_inputs=e['private_inputs_not_distributed']) for e in entries], indent=2))
        return
    selected = [entry for entry in entries if args.only is None or set(args.only) & set(entry['names'])]
    if args.output_dir.resolve() == (task_dir / 'figures').resolve():
        parser.error('Use a separate output directory to preserve the manuscript PDFs')
    for entry in selected:
        missing = [source for source in entry['private_inputs_not_distributed']
                   if not list(task_dir.glob(source))]
        if missing:
            parser.error('Authorized patient inputs required for ' + ', '.join(entry['names']) +
                         '; see DATA_ACCESS.md. Missing: ' + ', '.join(missing))
    records = []
    for entry in selected:
        context = ReplayContext(task_dir, entry['method'], args.output_dir)
        with manuscript_style(context, entry['names']) as record:
            functions[entry['function']](context, **entry.get('kwargs', {}))
        records.append(dict(figure=entry['supplementary_figure'], function=entry['function'],
            names=entry['names'], method=entry['method'], **record))
        print('REPLAYED', task_dir.name, ', '.join(entry['names']))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'replay_manifest.json').write_text(json.dumps(records, indent=2) + '\n')
