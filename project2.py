import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox
import re

# ————————————————————————————————————————————————————————————————
# Load abusive words from CSV (one word per line)
# ————————————————————————————————————————————————————————————————
try:
    abusive_df = pd.read_csv(
        r"C:\Users\lokes\Downloads\profanity_en.csv",
        header=None,
        names=["word"],
        encoding="utf-8",
    )
    abusive_words = (
        abusive_df["word"]
        .astype(str)
        .str.replace(r"^\ufeff", "", regex=True)  # strip BOM if present
        .str.strip()  # trim whitespace
        .str.lower()  # normalize case
        .tolist()
    )
    print(f"Loaded {len(abusive_words)} abusive words, sample:", abusive_words[:20])
except Exception as e:
    print(f"⚠ Could not load abusive‑words CSV: {e}")
    abusive_words = []

# Global variables for ML models & accuracies
vectorizer = None
lr_model = None
svm_model = None
rfc_model = None
lr_accuracy = None
svm_accuracy = None
rfc_accuracy = None


# ————————————————————————————————————————————————————————————————
def train_models():
    global vectorizer, lr_model, svm_model, rfc_model
    global lr_accuracy, svm_accuracy, rfc_accuracy

    try:
        data = pd.read_csv(r"C:\\Users\\lokes\\OneDrive\\Desktop\\MCA-AIML 2nd Sem\\ML\\twitter_racism_parsed_dataset.csv")
        data["text"] = data["text"].astype(str)

        vectorizer = TfidfVectorizer(
            max_features=2000, ngram_range=(1, 3), min_df=2, stop_words="english"
        )
        X = vectorizer.fit_transform(data["text"]).toarray()
        y = data["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        lr_model = LogisticRegression(
            solver="liblinear", C=1.0, class_weight="balanced"
        )
        lr_model.fit(X_train, y_train)
        lr_accuracy = accuracy_score(y_test, lr_model.predict(X_test))

        svm_model = SVC(
            kernel="linear", C=1.0, class_weight="balanced", probability=True
        )
        svm_model.fit(X_train, y_train)
        svm_accuracy = accuracy_score(y_test, svm_model.predict(X_test))

        rfc_model = RandomForestClassifier(
            n_estimators=200, max_depth=20, class_weight="balanced", random_state=42
        )
        rfc_model.fit(X_train, y_train)
        rfc_accuracy = accuracy_score(y_test, rfc_model.predict(X_test))

        messagebox.showinfo(
            "Training Complete",
            f"Models trained successfully!\n\n"
            f"Logistic Regression Accuracy: {lr_accuracy*100:.2f}%\n"
            f"SVM Accuracy:               {svm_accuracy*100:.2f}%\n"
            f"Random Forest Accuracy:     {rfc_accuracy*100:.2f}%",
        )
        predict_button.config(state="normal")

    except Exception as e:
        messagebox.showerror("Error", f"Training failed: {e}")


# ————————————————————————————————————————————————————————————————
def find_abusive_words(sentence):
    words = re.findall(r"\b\w+\b", sentence.lower())
    return [w for w in words if w in abusive_words]


def predict_abuse(sentence):
    abusive_found = find_abusive_words(sentence)
    keyword_flag = bool(abusive_found)

    input_vect = vectorizer.transform([sentence]).toarray()
    lr_pred = lr_model.predict(input_vect)[0]
    svm_pred = svm_model.predict(input_vect)[0]
    rfc_pred = rfc_model.predict(input_vect)[0]

    if keyword_flag:
        return {
            "Logistic Regression": "Abusive",
            "SVM": "Abusive",
            "Random Forest": "Abusive",
            "abusive_words": abusive_found,
            "keyword_detection": True,
        }
    else:
        return {
            "Logistic Regression": "Abusive" if lr_pred == 1 else "Non-abusive",
            "SVM": "Abusive" if svm_pred == 1 else "Non-abusive",
            "Random Forest": "Abusive" if rfc_pred == 1 else "Non-abusive",
            "abusive_words": [],
            "keyword_detection": False,
        }


def on_predict():
    sentence = entry.get().strip()
    if not sentence:
        messagebox.showwarning("Input Error", "Please enter a sentence.")
        return

    preds = predict_abuse(sentence)
    lines = [
        f"Logistic Regression: {preds['Logistic Regression']}",
        f"SVM:               {preds['SVM']}",
        f"Random Forest:     {preds['Random Forest']}",
    ]

    if preds["keyword_detection"]:
        lines.append("\n⚠ ABUSIVE CONTENT DETECTED")
        lines.append(f"Words found: {', '.join(preds['abusive_words'])}")
        lines.append("👉 Consider replacing with more respectful language.")
    elif any(
        r == "Abusive"
        for r in (preds["Logistic Regression"], preds["SVM"], preds["Random Forest"])
    ):
        flagged = [
            m
            for m, r in preds.items()
            if r == "Abusive" and m in ("Logistic Regression", "SVM", "Random Forest")
        ]
        lines.append(f"\n⚠ Flagged by: {', '.join(flagged)}")
    else:
        lines.append("\n✅ All models agree: Non-abusive")

    result_text.set("\n".join(lines))


def show_accuracy():
    if lr_model is None:
        messagebox.showwarning("Warning", "Please train the models first.")
    else:
        messagebox.showinfo(
            "Model Accuracies",
            f"Logistic Regression: {lr_accuracy*100:.2f}%\n"
            f"SVM:               {svm_accuracy*100:.2f}%\n"
            f"Random Forest:     {rfc_accuracy*100:.2f}%",
        )


# ————————————————————————————————————————————————————————————————
# Tkinter GUI
window = tk.Tk()
window.title("Abusive Text Classifier")
window.geometry("500x450")

tk.Label(window, text="Abusive Text Classifier", font=("Helvetica", 16, "bold")).pack(
    pady=10
)
tk.Button(
    window, text="Train Models", command=train_models, width=20, bg="blue", fg="white"
).pack(pady=5)

entry = tk.Entry(window, width=50, font=("Helvetica", 12))
entry.pack(pady=10)

predict_button = tk.Button(
    window,
    text="Predict",
    command=on_predict,
    width=15,
    bg="green",
    fg="white",
    state="disabled",
)
predict_button.pack(pady=5)

result_text = tk.StringVar()
tk.Label(
    window,
    textvariable=result_text,
    font=("Helvetica", 12),
    justify="left",
    wraplength=450,
).pack(pady=10)

tk.Button(window, text="Show Model Accuracies", command=show_accuracy, width=25).pack(
    pady=10
)

window.mainloop()