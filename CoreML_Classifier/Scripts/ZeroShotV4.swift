//
//  Untitled.swift
//  CoreML_Classifier
//
//  Created by Oktawian Głowacz on 15/05/2025.
//

import Foundation
import CoreML
import Vision
import AppKit
import UniformTypeIdentifiers

struct PredictionSample {
    var imagePath: String
    var groundTruth: String
    var prediction: String
}

struct Metrics {
    var accuracy: Double
    var precision: Double
    var recall: Double
    var f1Score: Double
}

struct ZeroShotClassifier {
    static func evaluateWithSavePanel(modelURL: URL, datasetURL: URL, batchSize: Int = 8) {
        let panel = NSSavePanel()
        panel.title = "Select Save Location for Zero-Shot Report"
        panel.nameFieldStringValue = "ZeroShot_Results"

        if let modelUTType = UTType(filenameExtension: "txt") {
            panel.allowedContentTypes = [modelUTType]
        }

        if panel.runModal() == .OK, let outputFile = panel.url {
            let outputFolder = outputFile.deletingLastPathComponent()
            evaluate(modelURL: modelURL, datasetURL: datasetURL, outputFolderURL: outputFolder, batchSize: batchSize)
        } else {
            print("❌ User cancelled zero-shot output folder selection")
        }
    }

    static func evaluate(modelURL: URL, datasetURL: URL, outputFolderURL: URL, batchSize: Int = 8) {
        do {
            let compiledURL = try MLModel.compileModel(at: modelURL)
            let model = try MLModel(contentsOf: compiledURL)
            let vnModel = try VNCoreMLModel(for: model)
            let modelName = modelURL.deletingPathExtension().lastPathComponent

            let fm = FileManager.default
            let classFolders = try fm.contentsOfDirectory(atPath: datasetURL.path)
                .filter { $0 != ".DS_Store" }

            var samples = [PredictionSample]()
            for className in classFolders {
                let classPath = datasetURL.appendingPathComponent(className).path
                let images = try fm.contentsOfDirectory(atPath: classPath)
                    .filter { $0.lowercased().hasSuffix("jpg") || $0.lowercased().hasSuffix("jpeg") || $0.lowercased().hasSuffix("png") }

                for image in images {
                    let imagePath = (classPath as NSString).appendingPathComponent(image)
                    samples.append(PredictionSample(imagePath: imagePath, groundTruth: className, prediction: ""))
                }
            }

            print("[🗺️] Found \(samples.count) images.")

            var groundTruths = [String]()
            var predictions = [String]()
            var updatedSamples = [PredictionSample]()

            for sample in samples {
                let url = URL(fileURLWithPath: sample.imagePath)
                guard let nsImage = NSImage(contentsOf: url),
                      let ciImage = CIImage(data: nsImage.tiffRepresentation ?? Data()) else {
                    print("⚠️ Failed to load image: \(url.path)")
                    continue
                }

                let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
                let request = VNCoreMLRequest(model: vnModel)
                try handler.perform([request])

                if let results = request.results as? [VNClassificationObservation],
                   let top = results.first {
                    print("🔍 GT: \(sample.groundTruth) | Pred: \(top.identifier)")
                    groundTruths.append(sample.groundTruth)
                    predictions.append(top.identifier)
                    updatedSamples.append(PredictionSample(imagePath: sample.imagePath, groundTruth: sample.groundTruth, prediction: top.identifier))
                }
            }

            let metrics = calculateMetrics(groundTruths: groundTruths, predictions: predictions)
            print("[📈] Accuracy: \(metrics.accuracy)")
            print("[📈] Precision: \(metrics.precision)")
            print("[📈] Recall: \(metrics.recall)")
            print("[📈] F1 Score: \(metrics.f1Score)")

            try? fm.createDirectory(at: outputFolderURL, withIntermediateDirectories: true)

            let classLabels = Array(Set(groundTruths + predictions)).sorted()
            let confusionMatrix = buildConfusionMatrix(groundTruths: groundTruths, predictions: predictions, classLabels: classLabels)

            saveConfusionMatrix(matrix: confusionMatrix,
                                classLabels: classLabels,
                                outputURL: outputFolderURL.appendingPathComponent("confusion_matrix_\(modelName).csv"))

            saveMetricsCSV(metrics: metrics, modelName: modelName, outputFolderURL: outputFolderURL)
            saveImagePredictionsCSV(samples: updatedSamples, modelName: modelName, outputFolderURL: outputFolderURL)

            print("[✅] Zero-shot evaluation completed.")

        } catch {
            print("❌ Error during evaluation: \(error)")
        }
    }

    static func saveMetricsCSV(metrics: Metrics, modelName: String, outputFolderURL: URL) {
        let csv = """
        Metric,Value
        Accuracy,\(metrics.accuracy)
        Precision,\(metrics.precision)
        Recall,\(metrics.recall)
        F1 Score,\(metrics.f1Score)
        """
        let metricsURL = outputFolderURL.appendingPathComponent("metrics_\(modelName).csv")
        try? csv.write(to: metricsURL, atomically: true, encoding: .utf8)
    }

    static func saveImagePredictionsCSV(samples: [PredictionSample], modelName: String, outputFolderURL: URL) {
        var csv = "Image,GroundTruth,Prediction\n"
        for sample in samples {
            let imageName = URL(fileURLWithPath: sample.imagePath).lastPathComponent
            csv += "\(imageName),\(sample.groundTruth),\(sample.prediction)\n"
        }
        let predictionsURL = outputFolderURL.appendingPathComponent("predictions_\(modelName).csv")
        try? csv.write(to: predictionsURL, atomically: true, encoding: .utf8)
    }

    static func calculateMetrics(groundTruths: [String], predictions: [String]) -> Metrics {
        let total = groundTruths.count
        let correct = zip(groundTruths, predictions).filter { $0 == $1 }.count
        let accuracy = Double(correct) / Double(total)

        var precision = 0.0, recall = 0.0, f1 = 0.0
        let labels = Array(Set(groundTruths + predictions))
        for label in labels {
            let tp = zip(groundTruths, predictions).filter { $0 == label && $1 == label }.count
            let fp = predictions.filter { $0 == label }.count - tp
            let fn = groundTruths.filter { $0 == label }.count - tp

            let p = Double(tp) / (Double(tp + fp) + 1e-10)
            let r = Double(tp) / (Double(tp + fn) + 1e-10)
            let f = 2 * p * r / (p + r + 1e-10)

            precision += p
            recall += r
            f1 += f
        }

        let count = Double(labels.count)
        return Metrics(accuracy: accuracy, precision: precision / count, recall: recall / count, f1Score: f1 / count)
    }
}
