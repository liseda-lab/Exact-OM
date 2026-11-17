"""Public source-sharded native evidence command."""
import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(prog='exact score', description=__doc__)
    parser.add_argument('-s', '--source', required=True)
    parser.add_argument('-t', '--target', required=True)
    parser.add_argument('-y', '--config', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--mode', choices=['global', 'local'], default='global')
    parser.add_argument('-c', '--candidates')
    parser.add_argument('--source-prefix', action='append', default=[])
    parser.add_argument('--target-prefix', action='append', default=[])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--sources-per-shard', type=int, default=64)
    args = parser.parse_args(argv)
    from exact.native.score import score
    score(args.source, args.target, args.config, args.output, candidates=args.candidates,
          mode=args.mode, device=args.device, source_prefixes=args.source_prefix,
          target_prefixes=args.target_prefix, sources_per_shard=args.sources_per_shard)
    return 0
