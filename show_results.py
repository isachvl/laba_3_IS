"""Build presentation plots and optionally import measured runs into MLflow."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
NAMES = {'dense': 'Dense', 'bm25': 'BM25', 'rrf': 'RRF', 'ce': 'RRF + CE'}
COLORS = {'dense': '#64748b', 'bm25': '#b45309', 'rrf': '#2563eb', 'ce': '#12805c'}


def plots(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    groups = [
        ('Поиск на dev', summary['retrieval']['dev'], ('hit', 'recall', 'rr'),
         ('Hit@3', 'Recall@3', 'MRR@3'), '01_dev_retrieval.png'),
        ('Ответы на holdout', summary['generation']['holdout'], ('accuracy', 'citation', 'evidence'),
         ('Точность', 'Верная ссылка', 'Обоснование в промпте'), '02_holdout_answers.png'),
    ]
    destination = RESULTS / 'plots'
    destination.mkdir(exist_ok=True)
    for title, methods, keys, labels, filename in groups:
        fig, ax = plt.subplots(figsize=(9, 4.3), layout='constrained')
        width = .78 / len(methods)
        for offset, (method, values) in enumerate(methods.items()):
            values = values['3'] if '3' in values else values
            x = [i + width*(offset-(len(methods)-1)/2) for i in range(len(keys))]
            bars = ax.bar(x, [100*values[key] for key in keys], width*.94,
                          label=NAMES[method], color=COLORS[method])
            ax.bar_label(bars, fmt='%.1f', fontsize=8, padding=3)
        ax.set(xticks=range(len(labels)), xticklabels=labels, ylim=(0, 110),
               ylabel='Доля, %', title=title)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        fig.legend(loc='outside lower center', ncol=len(methods), frameon=False)
        fig.savefig(destination / filename, dpi=160)
        plt.close(fig)


def import_mlflow(summary):
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri('sqlite:///' + (ROOT/'mlflow.db').as_posix())
    client = MlflowClient()
    experiment = client.get_experiment_by_name('lr3')
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(
        'lr3', artifact_location=(ROOT/'mlruns').as_uri())
    existing = {r.data.tags.get('lr3.case'): r for r in client.search_runs([experiment_id])}
    for split, methods in summary['generation'].items():
        for method, values in methods.items():
            case = f'{split}-{method}'
            prior = existing.get(case)
            metrics = {'accuracy': values['accuracy'], 'correct_citation': values['citation'],
                       'mrr_at_3': summary['retrieval'][split][method]['3']['rr']}
            with mlflow.start_run(run_id=prior.info.run_id if prior else None,
                                  experiment_id=experiment_id, run_name=f'{split} / {NAMES[method]}'):
                mlflow.set_tag('lr3.case', case)
                mlflow.log_params({'split': split, 'retriever': NAMES[method]})
                missing = {k:v for k,v in metrics.items() if prior is None or prior.data.metrics.get(k) != v}
                if missing:
                    mlflow.log_metrics(missing)
                mlflow.log_artifacts(str(RESULTS/'plots'), 'plots')
    print('MLflow: lr3, 5 runs, 3 metrics')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mlflow', action='store_true')
    args = parser.parse_args()
    summary = json.loads((RESULTS/'summary.json').read_text(encoding='utf-8'))
    plots(summary)
    if args.mlflow:
        import_mlflow(summary)

