from __future__ import annotations

from .runner import run_platoon_experiments


def main() -> int:
    output = run_platoon_experiments()
    if output.summary.empty:
        print("[platoon] no group evaluated. Check experiments.enabled_groups in TOML.")
        return 0

    print("[platoon] summary")
    print(output.summary.to_string(index=False))
    print(f"[platoon] outputs saved to: {output.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
