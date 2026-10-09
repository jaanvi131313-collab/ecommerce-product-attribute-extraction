"""Step 4 - Small GUI. Type a product description, click Extract, see the attributes.

Usage:  python gui.py        (needs the trained 'model' folder from train.py)
"""
import tkinter as tk
from tkinter import ttk, messagebox
import spacy
from common import MODEL_DIR

COLOURS = {"SLEEVE_STYLE": "#ffd6a5", "NECKLINE": "#caffbf", "LENGTH": "#9bf6ff", "PATTERN": "#ffc6ff"}

try:
    nlp = spacy.load(MODEL_DIR)
except Exception:
    nlp = None

root = tk.Tk()
root.title("E-commerce Attribute Extractor")
root.geometry("640x520")

ttk.Label(root, text="Type a clothing product description:", font=("Segoe UI", 11)).pack(anchor="w", padx=12, pady=(12, 4))
box = tk.Text(root, height=4, wrap="word", font=("Segoe UI", 11))
box.pack(fill="x", padx=12)
box.insert("1.0", "Women's black floral maxi dress with short sleeves and a V-neck")

ttk.Label(root, text="Highlighted result:", font=("Segoe UI", 11)).pack(anchor="w", padx=12, pady=(12, 4))
out = tk.Text(root, height=4, wrap="word", font=("Segoe UI", 11), state="disabled", bg="#f6f6f6")
out.pack(fill="x", padx=12)

table = ttk.Treeview(root, columns=("attr", "value"), show="headings", height=8)
table.heading("attr", text="Attribute")
table.heading("value", text="Extracted value")
table.column("attr", width=160)
table.column("value", width=400)
table.pack(fill="both", expand=True, padx=12, pady=12)

for label, colour in COLOURS.items():
    out.tag_configure(label, background=colour)


def extract(event=None):
    if nlp is None:
        messagebox.showerror("Model not found", "Run train.py first so the 'model' folder exists.")
        return "break"
    text = box.get("1.0", "end").strip()
    if not text:
        return "break"
    doc = nlp(text)
    table.delete(*table.get_children())
    out.config(state="normal")
    out.delete("1.0", "end")
    out.insert("1.0", text)
    for ent in doc.ents:
        out.tag_add(ent.label_, f"1.0+{ent.start_char}c", f"1.0+{ent.end_char}c")
        table.insert("", "end", values=(ent.label_, ent.text))
    if not doc.ents:
        table.insert("", "end", values=("-", "No attributes found"))
    out.config(state="disabled")
    return "break"


ttk.Button(root, text="Extract attributes", command=extract).pack(pady=(0, 12))
box.bind("<Return>", extract)          # Enter also extracts
root.mainloop()
