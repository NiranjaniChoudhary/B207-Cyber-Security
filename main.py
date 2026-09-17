
import argparse
import sys
from pathlib import Path

from model import PhishingDetectionModel, DEFAULT_MODEL_PATH
from database import Database
from preprocess import load_email_from_file


def cmd_train(args):
    # Delegate to train_model.py's logic to avoid duplicating code.
    from train_model import main as train_main
    sys.argv = ["train_model.py"]
    if args.data:
        sys.argv += ["--data", args.data]
    if args.algorithm:
        sys.argv += ["--algorithm", args.algorithm]
    train_main()


def _load_model():
    model_path = Path(DEFAULT_MODEL_PATH)
    if not model_path.exists():
        print("[!] No trained model found. Run `python main.py train` first.")
        sys.exit(1)
    return PhishingDetectionModel.load(model_path)


def cmd_analyze(args):
    if args.file:
        parsed = load_email_from_file(args.file)
        subject, body = parsed["subject"], parsed["body"]
        source = f"file:{args.file}"
    elif args.text:
        subject, body = "", args.text
        source = "cli-text"
    elif args.body:
        subject, body = (args.subject or ""), args.body
        source = "cli-fields"
    else:
        print("[!] Provide --file, --text, or --subject/--body.")
        sys.exit(1)

    model = _load_model()
    label, confidence, feature_dict = model.predict(subject, body)

    db = Database()
    email_id = db.add_email(body=body, subject=subject, source=source)
    db.add_analysis_result(
        email_id=email_id,
        prediction=label,
        confidence=confidence,
        features=feature_dict,
        model_version=getattr(model, "version", None),
    )

    print("\n=== Phishing Analysis Report ===")
    print(f"Subject      : {subject!r}")
    print(f"Prediction   : {label.upper()}")
    print(f"Confidence   : {confidence * 100:.2f}%")
    print("\nExtracted indicators:")
    for key, value in feature_dict.items():
        print(f"  - {key}: {value}")

    if label == "phishing":
        print("\n[!] WARNING: This email shows strong indicators of a phishing attempt.")
        print("    Do not click any links or provide personal information.")
    else:
        print("\n[OK] No strong phishing indicators detected.")
        print("    (Always remain cautious with unexpected requests for personal info.)")


def cmd_list(args):
    db = Database()
    results = db.get_recent_results(limit=args.limit)
    if not results:
        print("No analysis results yet. Run `python main.py analyze ...` first.")
        return
    print(f"{'ID':<5}{'Subject':<40}{'Prediction':<12}{'Confidence':<12}{'Analyzed At'}")
    print("-" * 100)
    for r in results:
        subject = (r["subject"] or "")[:38]
        print(f"{r['id']:<5}{subject:<40}{r['prediction']:<12}"
              f"{r['confidence']*100:>6.2f}%     {r['analyzed_at']}")


def cmd_phishing(args):
    db = Database()
    attempts = db.get_phishing_attempts(limit=args.limit)
    if not attempts:
        print("No phishing attempts recorded yet.")
        return
    print(f"{'ID':<5}{'Subject':<40}{'Confidence':<12}{'Detected At'}")
    print("-" * 90)
    for a in attempts:
        subject = (a["subject"] or "")[:38]
        print(f"{a['id']:<5}{subject:<40}{a['confidence']*100:>6.2f}%     {a['detected_at']}")


def cmd_stats(args):
    db = Database()
    stats = db.get_stats()
    print("=== Phishing Detection System - Statistics ===")
    print(f"Total emails analyzed : {stats['total_analyzed']}")
    print(f"Phishing detected     : {stats['phishing_detected']}")
    print(f"Legitimate            : {stats['legitimate']}")
    print(f"Average confidence    : {stats['average_confidence']}")
    if stats["latest_model"]:
        m = stats["latest_model"]
        print("\nLatest trained model:")
        print(f"  Version   : {m['model_version']}")
        print(f"  Algorithm : {m['algorithm']}")
        print(f"  Accuracy  : {m['accuracy']}")
        print(f"  Precision : {m['precision_score']}")
        print(f"  Recall    : {m['recall_score']}")
        print(f"  F1 Score  : {m['f1_score']}")
    else:
        print("\nNo model metadata found. Run `python main.py train` first.")


def cmd_report(args):
    db = Database()
    stats = db.get_stats()
    results = db.get_recent_results(limit=args.limit)
    attempts = db.get_phishing_attempts(limit=args.limit)

    lines = []
    lines.append("PHISHING EMAIL DETECTION SYSTEM - REPORT")
    lines.append("=" * 50)
    lines.append(f"Total emails analyzed : {stats['total_analyzed']}")
    lines.append(f"Phishing detected     : {stats['phishing_detected']}")
    lines.append(f"Legitimate            : {stats['legitimate']}")
    lines.append(f"Average confidence    : {stats['average_confidence']}")
    lines.append("")
    lines.append("Recent phishing attempts:")
    lines.append("-" * 50)
    for a in attempts:
        lines.append(f"[{a['detected_at']}] {a['subject']} "
                      f"(confidence: {a['confidence']*100:.2f}%)")
    lines.append("")
    lines.append("Recent analysis activity:")
    lines.append("-" * 50)
    for r in results:
        lines.append(f"[{r['analyzed_at']}] {r['subject']} -> "
                      f"{r['prediction']} ({r['confidence']*100:.2f}%)")

    report_text = "\n".join(lines)
    if args.output:
        Path(args.output).write_text(report_text, encoding="utf-8")
        print(f"[*] Report written to {args.output}")
    else:
        print(report_text)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="phishing-detector",
        description="Phishing Email Detection System (ML + SQLite)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train the ML model")
    p_train.add_argument("--data", help="Path to labeled CSV dataset")
    p_train.add_argument("--algorithm", choices=["random_forest", "logistic_regression"])
    p_train.set_defaults(func=cmd_train)

    p_analyze = sub.add_parser("analyze", help="Analyze an email")
    p_analyze.add_argument("--file", help="Path to a .eml or .txt email file")
    p_analyze.add_argument("--text", help="Raw email text (subject+body combined)")
    p_analyze.add_argument("--subject", help="Email subject (used with --body)")
    p_analyze.add_argument("--body", help="Email body (used with --subject)")
    p_analyze.set_defaults(func=cmd_analyze)

    p_list = sub.add_parser("list", help="List recent analysis results")
    p_list.add_argument("--limit", type=int, default=20)
    p_list.set_defaults(func=cmd_list)

    p_phishing = sub.add_parser("phishing", help="List detected phishing attempts")
    p_phishing.add_argument("--limit", type=int, default=50)
    p_phishing.set_defaults(func=cmd_phishing)

    p_stats = sub.add_parser("stats", help="Show detection statistics")
    p_stats.set_defaults(func=cmd_stats)

    p_report = sub.add_parser("report", help="Generate a text report")
    p_report.add_argument("--output", help="Write report to this file instead of stdout")
    p_report.add_argument("--limit", type=int, default=20)
    p_report.set_defaults(func=cmd_report)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
