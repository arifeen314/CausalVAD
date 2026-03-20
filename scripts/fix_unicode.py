#!/usr/bin/env python3
"""
Quick patch: fixes the Unicode star character that crashes Windows Command Prompt.
Run this once:  python scripts/fix_unicode.py
"""
from pathlib import Path

trainer_file = Path(__file__).parent.parent / "src" / "training" / "trainer.py"
content = trainer_file.read_text(encoding="utf-8")

# Replace the Unicode star with a plain ASCII version
content = content.replace(
    '★ New best AUC:',
    '[BEST] New best AUC:'
)
content = content.replace(
    '— checkpoint saved',
    '- checkpoint saved'
)

trainer_file.write_text(content, encoding="utf-8")
print("Fixed Unicode characters in trainer.py")
print("The logging error will not appear again.")
