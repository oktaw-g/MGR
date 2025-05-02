# 📦 Zero-Shot Image Classifier Evaluator (CreateML + CoreML)

This project lets you **evaluate** any pre-trained CoreML model (`.mlmodel` or `.mlpackage`) on a **custom dataset**, calculating important metrics like accuracy, precision, recall, and F1 score — and generates an **HTML report** with confusion matrix and random sample visualizations!

---

## 📂 Project Structure

```
/path/to/your/models/
    ├── model1.mlmodel
    ├── model2.mlmodel
    └── model3.mlpackage

/path/to/your/dataset/
    ├── ClassA/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── ClassB/
    │   ├── image1.jpg
    │   └── ...
    └── ...

/path/to/output/folder/
    └── (Generated reports, confusion matrix CSVs, sample images)
```

---

## ⚙️ Requirements

- macOS (Monterey or newer recommended)
- Xcode with CreateML and CoreML frameworks installed
- Swift 5.7+ project (macOS Command Line App template works best)

---

## 🚀 How to Use

1. Open **Xcode** and create a **Command Line macOS App** project.
2. Add the provided `main.swift` code into your project.
3. Update the following paths at the top of the script:
   ```swift
   let modelFolderPath = "/path/to/your/models"
   let datasetFolderPath = "/path/to/your/dataset"
   let outputFolderPath = "/path/to/output/folder"
   ```
4. Build and Run the app.
5. The script will display a list of models.  
   **Enter the number** of the model you want to evaluate.

6. After evaluation, you will find:
   - `confusion_matrix.csv`
   - random sample images with predictions
   - a complete `report.html` file in your output folder.

---

## 🛠️ Configurable Options

- **Batch Size**  
  You can adjust how many images are evaluated in parallel by changing:
  ```swift
  let batchSize = 8
  ```

- **Number of Samples Shown**  
  You can control how many random samples are included in the HTML report:
  ```swift
  saveRandomPredictions(samples: evaluatedSamples, outputFolder: samplesFolder, count: 3)
  ```

---

## 📈 Metrics Calculated

- Accuracy
- Precision
- Recall
- F1 Score

All based on your provided dataset and model predictions!

---

## 📑 Example Outputs

After evaluation, you will have:

- ✅ An **HTML report** summarizing metrics
- ✅ A **Confusion Matrix** as `.csv`
- ✅ **3 random images** showing true and predicted labels

---

## 🎯 Goal

The goal is to **compare** different models **before and after training** on your dataset using consistent metrics and sample visualizations.

---

# 🔥 Happy Evaluating!


