"""Fit and apply the final global acceptance stage to scored candidate tables."""

import argparse


def main(argv=None):
    """Run explicit TRAIN fitting or label-free application of a frozen artifact."""
    parser = argparse.ArgumentParser(prog="exact acceptance", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser(
        "fit", help="Fit on public TRAIN; optionally qualify assignment on VALID"
    )
    fit.add_argument(
        "--candidates", required=True, help="Complete scored candidate CSV/TSV/parquet"
    )
    fit.add_argument("--train", required=True, help="Public positive TRAIN pair table")
    fit.add_argument("--valid", help="Optional public positive VALID table; never TEST")
    fit.add_argument("--output", required=True)
    fit.add_argument("--family", choices=["hist", "logistic"], default="hist")
    fit.add_argument("--weak-weight", type=float, default=0.02)
    fit.add_argument("--folds", type=int, default=5)
    fit.add_argument("--seed", type=int, default=20261116)
    fit.add_argument("--background-size", type=int, default=8192)
    fit.add_argument("--fold-membership", help="Optional JSON source-to-fold map")
    fit.add_argument("--threads", type=int, default=1)
    apply = commands.add_parser(
        "apply", help="Apply a trusted frozen model without reference labels"
    )
    apply.add_argument("--candidates", required=True)
    apply.add_argument("--model", required=True, help="Trusted model bundle directory")
    apply.add_argument("--policy", choices=["full", "partial"], required=True)
    apply.add_argument("--output", required=True)
    apply.add_argument("--threads", type=int, default=1)
    imported = commands.add_parser(
        "import-model", help="Wrap a trusted frozen sklearn model without refitting"
    )
    imported.add_argument("--model", required=True, help="Trusted (scaler, estimator) joblib file")
    imported.add_argument("--output", required=True)
    imported.add_argument("--threshold", type=float, required=True)
    imported.add_argument("--mode", choices=["greedy", "winner", "assignment"], required=True)
    imported.add_argument("--policy", choices=["full", "partial"], required=True)
    imported.add_argument(
        "--trusted-model",
        action="store_true",
        help="Acknowledge the model file uses trusted pickle serialization",
    )
    retained = commands.add_parser(
        "import-retained", help="Import a frozen numeric rank/winner head without refitting"
    )
    retained.add_argument("--head", required=True, help="Frozen corrected selector JSON")
    retained.add_argument("--temperature", type=float, required=True)
    retained.add_argument("--output", required=True)
    retained_apply = commands.add_parser(
        "apply-retained", help="Apply a frozen rank/winner head with its calibrated score floor"
    )
    retained_apply.add_argument("--candidates", required=True)
    retained_apply.add_argument("--model", required=True)
    retained_apply.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    from exact.acceptance import model

    try:
        if args.command == "fit":
            result = model.fit(
                args.candidates,
                args.train,
                args.output,
                valid_path=args.valid,
                family=args.family,
                weak_weight=args.weak_weight,
                folds=args.folds,
                seed=args.seed,
                background_size=args.background_size,
                threads=args.threads,
                membership_path=args.fold_membership,
            )
        elif args.command == "apply":
            result = model.apply(
                args.candidates, args.model, args.output, policy=args.policy, threads=args.threads
            )
        elif args.command == "import-model":
            result = model.import_model(
                args.model,
                args.output,
                threshold=args.threshold,
                mode=args.mode,
                policy=args.policy,
                trusted=args.trusted_model,
            )
        elif args.command == "import-retained":
            from exact.acceptance.retained import import_head

            result = import_head(args.head, args.output, temperature=args.temperature)
        else:
            from exact.acceptance.retained import apply

            result = apply(args.candidates, args.model, args.output)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
