"""Run T2E from the dataset to final predictions with one command."""
import argparse
import json
from datetime import datetime
from pathlib import Path
from t2e.pipeline import run
from t2e.configuration import load_config


def main():
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('ICEWS14','ICEWS05-15','ICEWS18'), default='ICEWS14')
    parser.add_argument('--model-path', help='Existing local InternLM2 model directory')
    parser.add_argument('--config', help='Alternative JSON configuration file')
    parser.add_argument('--window', type=int, help='Transaction witness window W in days (default: config.json)')
    parser.add_argument('--tau', help='Agreement threshold in (0, 1] (default: config.json)')
    parser.add_argument('--packet-cap', metavar='K', help='Per-candidate packet cap: positive integer or unlimited')
    parser.add_argument('--symbolic-weight', type=float, help='Symbolic calibration coefficient (default: 0.3)')
    parser.add_argument('--print-config', action='store_true', help='Print effective settings without loading data or model')
    parser.add_argument('--output', help='New run directory; same directory resumes identical inputs')
    parser.add_argument('--prepare-only', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        config = load_config(project, args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    model = Path(args.model_path or config['model_path']).expanduser()
    model = (model if model.is_absolute() or args.model_path else project / model).resolve()
    if args.print_config:
        print(json.dumps(dict(dataset=args.dataset, model_path_resolved=str(model), config=config), indent=2))
        return
    if not args.output:
        args.output = str(project/'outputs'/(args.dataset+'-'+datetime.now().strftime('%Y%m%d-%H%M%S')))
    run(args, project, config=config)


if __name__ == '__main__':
    main()
