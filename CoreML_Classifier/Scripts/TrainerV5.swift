//
//  TrainerV5.swift
//  CoreML_Classifier
//
//  Created by Oktawian Głowacz on 15/05/2025.
//

import CreateML
import Foundation
import AppKit
import CoreML
import Vision
import UniformTypeIdentifiers



import CreateML
import Foundation
import AppKit
import CoreML
import Vision
import UniformTypeIdentifiers



struct Trainer {
    static func train(baseModelURL: URL, datasetURL: URL, outputFolder: URL) -> MLImageClassifier? {
        do {
            let trainFolder = datasetURL.appendingPathComponent("train")
            let valFolder = datasetURL.appendingPathComponent("val")

            print("[📂] Training from: \(trainFolder.path)")
            print("[📂] Validation from: \(valFolder.path)")

            let trainingData = try MLImageClassifier.DataSource.labeledDirectories(at: trainFolder)
            let validationData = try MLImageClassifier.DataSource.labeledDirectories(at: valFolder)

            let parameters = MLImageClassifier.ModelParameters(
                validationData: validationData,
                maxIterations: 20,
                augmentationOptions: [.flip, .exposure, .blur]
            )

            let model = try MLImageClassifier(trainingData: trainingData, parameters: parameters)

            print("[⚙️] Training complete.")

            let testFolder = datasetURL.appendingPathComponent("test")
            saveModelWithDialog(model: model, baseModelURL: baseModelURL, testFolder: testFolder)

           
            return model

        } catch {
            print("❌ Training failed: \(error.localizedDescription)")
            return nil
        }
    }


    static func evaluateModel(model: MLModel, testFolder: URL) -> (Metrics, [[Int]], [String]) {
        var groundTruths = [String]()
        var predictions = [String]()

        do {
            let vnModel = try VNCoreMLModel(for: model)
            let fileManager = FileManager.default

            let classFolders = try fileManager.contentsOfDirectory(at: testFolder, includingPropertiesForKeys: nil, options: .skipsHiddenFiles)
            for classFolder in classFolders {
                guard classFolder.hasDirectoryPath else { continue }

                let className = classFolder.lastPathComponent
                let images = try fileManager.contentsOfDirectory(at: classFolder, includingPropertiesForKeys: nil, options: .skipsHiddenFiles)

                for imgPath in images {
                    let fullPath = classFolder.appendingPathComponent(imgPath.lastPathComponent)
                    if let nsImage = NSImage(contentsOf: fullPath),
                       let ciImage = CIImage(data: nsImage.tiffRepresentation ?? Data()) {

                        let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
                        let request = VNCoreMLRequest(model: vnModel)
                        try handler.perform([request])

                        if let results = request.results as? [VNClassificationObservation],
                           let top = results.first {
                            groundTruths.append(className)
                            predictions.append(top.identifier)
                        }
                    }
                }
            }
        } catch {
            print("❌ Evaluation failed: \(error.localizedDescription)")
        }

        let labels = Array(Set(groundTruths + predictions)).sorted()
        var labelIndex = [String: Int]()
        for (i, label) in labels.enumerated() {
            labelIndex[label] = i
        }

        var matrix = Array(repeating: Array(repeating: 0, count: labels.count), count: labels.count)
        var correct = 0
        for (gt, pred) in zip(groundTruths, predictions) {
            if gt == pred { correct += 1 }
            if let gtIdx = labelIndex[gt], let predIdx = labelIndex[pred] {
                matrix[gtIdx][predIdx] += 1
            }
        }

        let total = groundTruths.count
        let accuracy = Double(correct) / Double(total)

        var precision = 0.0, recall = 0.0, f1 = 0.0
        for i in 0..<labels.count {
            let tp = matrix[i][i]
            let fp = (0..<labels.count).map { matrix[$0][i] }.reduce(0, +) - tp
            let fn = (0..<labels.count).map { matrix[i][$0] }.reduce(0, +) - tp

            let p = Double(tp) / (Double(tp + fp) + 1e-10)
            let r = Double(tp) / (Double(tp + fn) + 1e-10)
            let f = 2 * p * r / (p + r + 1e-10)

            precision += p
            recall += r
            f1 += f
        }

        let count = Double(labels.count)
        let metrics = Metrics(accuracy: accuracy, precision: precision / count, recall: recall / count, f1Score: f1 / count)

        return (metrics, matrix, labels)
    }

    static func saveConfusionMatrix(matrix: [[Int]], classLabels: [String], outputURL: URL) {
        var csv = "," + classLabels.joined(separator: ",") + "\n"
        for (i, row) in matrix.enumerated() {
            let line = [classLabels[i]] + row.map { String($0) }
            csv += line.joined(separator: ",") + "\n"
        }
        try? csv.write(to: outputURL, atomically: true, encoding: .utf8)
    }

    static func saveMetricsCSV(metrics: Metrics, modelName: String, outputFolderURL: URL) {
        let metricsURL = outputFolderURL.appendingPathComponent("metrics_trained_\(modelName).csv")

        let csv = """
        Metric,Value
        Accuracy,\(metrics.accuracy)
        Precision,\(metrics.precision)
        Recall,\(metrics.recall)
        F1 Score,\(metrics.f1Score)
        """

        print("📄 Saving metrics to: \(metricsURL.path)")
        try? csv.write(to: metricsURL, atomically: true, encoding: .utf8)
    }


    static func saveModelWithDialog(model: MLImageClassifier, baseModelURL: URL, testFolder: URL) {
        let baseModelName = baseModelURL.deletingPathExtension().lastPathComponent

        let panel = NSSavePanel()
        panel.title = "Save Trained Model"
        panel.nameFieldStringValue = "TrainedClassifier_\(baseModelName).mlmodel"

        if let modelUTType = UTType(filenameExtension: "mlmodel") {
            panel.allowedContentTypes = [modelUTType]
        }

        if panel.runModal() == .OK, let saveURL = panel.url {
            do {
                if FileManager.default.fileExists(atPath: saveURL.path) {
                    try FileManager.default.removeItem(at: saveURL)
                }
                try model.write(to: saveURL)
                print("✅ Model saved to: \(saveURL.path)")

                let outputFolder = saveURL.deletingLastPathComponent()
                let compiledURL = try MLModel.compileModel(at: saveURL)
                let mlmodel = try MLModel(contentsOf: compiledURL)
                let (metrics, matrix, labels) = evaluateModel(model: mlmodel, testFolder: testFolder)

                saveConfusionMatrix(matrix: matrix, classLabels: labels, outputURL: outputFolder.appendingPathComponent("confusion_matrix_trained_\(baseModelName).csv"))
                saveMetricsCSV(metrics: metrics, modelName: baseModelName, outputFolderURL: outputFolder)
                // saveTrainingMetricsCSV(...) skipped due to missing metric details in current CreateML version

                print("✅ Evaluation complete. Metrics and confusion matrix saved.")
            } catch {
                print("❌ Error during model save or evaluation: \(error.localizedDescription)")
            }
        } else {
            print("❌ User cancelled save panel.")
        }
    }
}
