from pathlib import Path
import argparse
import importlib
import json
from .pipeline import AlzheimerAgent


def load_backend(entry):
    module, name = entry.split(':', 1)
    return getattr(importlib.import_module(module), name)()


def write_result(result, output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    report = result['report']
    (out/'clinician_report.md').write_text(report['clinician_report'])
    (out/'patient_summary.md').write_text(report['patient_summary'])
    (out/'technical_appendix.json').write_text(json.dumps(report['technical_appendix'], ensure_ascii=False, indent=2))
    print(str(out.resolve()))


def main():
    parser = argparse.ArgumentParser(description='AlzheimerAgent paper-aligned source release')
    sub = parser.add_subparsers(dest='command', required=True)
    demo = sub.add_parser('demo'); demo.add_argument('--output', required=True)
    run = sub.add_parser('run'); run.add_argument('--case', required=True); run.add_argument('--backend', required=True)
    run.add_argument('--question', action='append', default=[]); run.add_argument('--output', required=True)
    train = sub.add_parser('train'); train.add_argument('--manifest', required=True); train.add_argument('--spec', required=True)
    train.add_argument('--output', required=True); train.add_argument('--device', default='cpu'); train.add_argument('--seed', type=int, default=3407)
    prep = sub.add_parser('preprocess'); prep.add_argument('--mri', required=True); prep.add_argument('--pet')
    prep.add_argument('--template', required=True); prep.add_argument('--brain-age-reference', required=True)
    prep.add_argument('--output', required=True); prep.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.command == 'demo':
        from .demo import SyntheticBackend, synthetic_case
        result = AlzheimerAgent(SyntheticBackend()).run(synthetic_case(), ['What is the recorded volume?'])
        result['synthetic_demo'] = True
        write_result(result, args.output)
    elif args.command == 'run':
        result = AlzheimerAgent(load_backend(args.backend)).run(json.loads(Path(args.case).read_text()), args.question)
        write_result(result, args.output)
    elif args.command == 'train':
        from .training import train, TrainingConfig
        train(args.manifest, json.loads(Path(args.spec).read_text()), args.output,
              config=TrainingConfig(seed=args.seed), device=args.device)
    elif args.command == 'preprocess':
        from .preprocessing import preprocessing_plan, execute_plan
        commands = preprocessing_plan(args.mri, args.template, args.brain_age_reference, args.output, args.pet)
        print(json.dumps(execute_plan(commands, args.output) if args.execute else commands, indent=2))

if __name__ == '__main__': main()
