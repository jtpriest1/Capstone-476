"""CLI entry point for local development.

Usage:
  python algoml.py serve          # Start the API server
  python algoml.py train          # Train a model on a local dataset
  python algoml.py predict <text> # Quick single-message prediction from the command line
"""
import sys
import argparse


def cmd_serve(args):
    import uvicorn
    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)


def cmd_train(args):
    from app.config import settings
    from app.main import run_training
    from app.schemas import TrainRequest, ColumnMapOverride

    col_map = None
    if args.text_col or args.label_col or args.pos_value:
        col_map = ColumnMapOverride(
            text_column=args.text_col or settings.dataset_text_column,
            label_column=args.label_col or settings.dataset_label_column,
            label_positive_value=args.pos_value or settings.dataset_label_positive_value,
        )

    request = TrainRequest(
        model_name=args.model,
        dataset_path=args.dataset,
        dataset_format=args.format,
        column_map=col_map,
    )
    result = run_training(request, settings)
    print(f"\nJob:    {result.job_id}")
    print(f"Status: {result.status}")
    if result.metrics:
        m = result.metrics
        print(f"Accuracy:  {m.accuracy:.4f}")
        print(f"Precision: {m.precision:.4f}")
        print(f"Recall:    {m.recall:.4f}")
        print(f"F1:        {m.f1:.4f}")
        if m.auc_roc:
            print(f"AUC-ROC:   {m.auc_roc:.4f}")
    if result.error:
        print(f"Error: {result.error}", file=sys.stderr)


def cmd_predict(args):
    from app.config import settings
    from app.model import get_loaded, try_load_from_disk
    from app.data import clean_text

    try_load_from_disk(settings.model_dir)

    model_name = args.model or settings.default_model_name
    try:
        detector = get_loaded(model_name)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    text = clean_text(args.text)
    is_scam, confidence = detector.predict(text)
    explanation = detector.explain(text)

    print(f"\nInput:      {args.text[:80]}")
    print(f"Is scam:    {is_scam}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Top tokens: {explanation[:5]}")


def main():
    parser = argparse.ArgumentParser(description="Scam Detection ML Backend")
    sub = parser.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("serve", help="Start the FastAPI server")
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--reload", action="store_true", default=False)

    tp = sub.add_parser("train", help="Train a model on a local dataset")
    tp.add_argument("--model", default="tfidf_logreg")
    tp.add_argument("--dataset", required=True, help="Filename inside datasets/ dir")
    tp.add_argument("--format", default="csv", choices=["csv", "json"])
    tp.add_argument("--text-col", default=None, help="Column name for text")
    tp.add_argument("--label-col", default=None, help="Column name for labels")
    tp.add_argument("--pos-value", default=None, help="Label value that means scam/spam")

    pp = sub.add_parser("predict", help="Run a single prediction from the CLI")
    pp.add_argument("text", help="The message text to classify")
    pp.add_argument("--model", default=None)

    args = parser.parse_args()
    {"serve": cmd_serve, "train": cmd_train, "predict": cmd_predict}[args.command](args)


if __name__ == "__main__":
    main()
