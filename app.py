from flask import Flask, render_template, request
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
import io, base64

app = Flask(__name__)

# Muat model Decision Tree
model = joblib.load("model_decision_tree.pkl")

# Daftar fitur
features = ['Age', 'Gender', 'Daily_Screen_Time(hrs)', 
            'Sleep_Quality(1-10)', 'Stress_Level(1-10)',
            'Days_Without_Social_Media', 'Exercise_Frequency(week)',
            'Social_Media_Platform']

# Baca dataset untuk evaluasi Confusion Matrix & ROC
df = pd.read_csv("Mental_Health_and_Social_Media_Balance_Dataset.csv")

# Buat kolom target kategorikal
def happiness_category(x):
    if x <= 4:
        return "Low"
    elif x <= 7:
        return "Medium"
    else:
        return "High"

df['Happiness_Level'] = df['Happiness_Index(1-10)'].apply(happiness_category)
X = df.drop(columns=['User_ID', 'Happiness_Index(1-10)', 'Happiness_Level'])
y = df['Happiness_Level']

# Pastikan encoding sama seperti saat training
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = pd.factorize(X[col])[0]

# Prediksi untuk confusion matrix & ROC
y_pred = model.predict(X)
y_pred_proba = model.predict_proba(X)

# Konversi label ke format numerik
classes = model.classes_
y_bin = label_binarize(y, classes=classes)

# Buat confusion matrix global
cm = confusion_matrix(y, y_pred, labels=classes)

# Fungsi bantu untuk ubah plot jadi base64
def fig_to_base64():
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()
    return encoded

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data input
        input_data = [float(request.form[feat]) for feat in features]
        df_input = pd.DataFrame([input_data], columns=features)

        # Prediksi & probabilitas
        prediction = model.predict(df_input)[0]
        proba = model.predict_proba(df_input)[0]

        # ---------- GRAFIK PROBABILITAS ----------
        plt.figure(figsize=(5,3))
        plt.bar(classes, proba, color=['#ff7675', '#74b9ff', '#55efc4'])
        plt.title('Probabilitas Setiap Kelas Kebahagiaan')
        plt.xlabel('Kategori')
        plt.ylabel('Probabilitas')
        plt.ylim(0,1)
        chart_img = fig_to_base64()

        # ---------- CONFUSION MATRIX ----------
        plt.figure(figsize=(4,3))
        sns.heatmap(cm, annot=True, cmap='Greens', fmt='d',
                    xticklabels=classes, yticklabels=classes)
        plt.title("Confusion Matrix Model Decision Tree")
        plt.xlabel("Prediksi")
        plt.ylabel("Aktual")
        cm_img = fig_to_base64()

        # ---------- ROC CURVE ----------
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(len(classes)):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(5,4))
        colors = ['#ff7675', '#74b9ff', '#55efc4']
        for i, color in zip(range(len(classes)), colors):
            plt.plot(fpr[i], tpr[i], color=color,
                     label=f"{classes[i]} (AUC = {roc_auc[i]:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Kurva ROC Multi-Kelas')
        plt.legend(loc="lower right")
        roc_img = fig_to_base64()

        # Kirim ke template
        return render_template('result.html',
                               result=prediction,
                               chart_url=chart_img,
                               cm_url=cm_img,
                               roc_url=roc_img)

    except Exception as e:
        return render_template('result.html', result=f"Error: {e}",
                               chart_url=None, cm_url=None, roc_url=None)

if __name__ == '__main__':
    app.run(debug=True)
