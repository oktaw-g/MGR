//
//  ZeroShotClassifierV3.swift
//  CoreML_Classifier
//
//  Created by Oktawian Głowacz on 18/04/2025.
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

            saveConfusionMatrix(matrix: confusionMatrix, classLabels: classLabels, outputURL: outputFolderURL.appendingPathComponent("confusion_matrix.csv"))
            saveRandomPredictions(samples: updatedSamples, outputFolder: outputFolderURL.appendingPathComponent("samples"), count: 3)
            generateHTMLReport(metrics: metrics, confusionMatrixURL: outputFolderURL.appendingPathComponent("confusion_matrix.csv"), sampleFolderURL: outputFolderURL.appendingPathComponent("samples"), outputHTMLURL: outputFolderURL.appendingPathComponent("report.html"))

            print("[✅] Zero-shot evaluation completed.")

        } catch {
            print("❌ Error during evaluation: \(error)")
        }
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



    static func buildConfusionMatrix(groundTruths: [String], predictions: [String], classLabels: [String]) -> [[Int]] {
        var classToIndex = [String: Int]()
        for (index, label) in classLabels.enumerated() {
            classToIndex[label] = index
        }
        
        var matrix = Array(repeating: Array(repeating: 0, count: classLabels.count), count: classLabels.count)
        
        for (gt, pred) in zip(groundTruths, predictions) {
            if let gtIndex = classToIndex[gt], let predIndex = classToIndex[pred] {
                matrix[gtIndex][predIndex] += 1
            }
        }
        
        return matrix
    }

    static func saveConfusionMatrix(matrix: [[Int]], classLabels: [String], outputURL: URL) {
        var csv = "GroundTruth/Predicted," + classLabels.joined(separator: ",") + "\n"
        for (i, row) in matrix.enumerated() {
            csv += classLabels[i] + "," + row.map { String($0) }.joined(separator: ",") + "\n"
        }
        try? csv.write(to: outputURL, atomically: true, encoding: .utf8)
    }

    static func saveRandomPredictions(samples: [PredictionSample], outputFolder: URL, count: Int = 3) {
        try? FileManager.default.createDirectory(at: outputFolder, withIntermediateDirectories: true)
        let selectedSamples = samples.shuffled().prefix(count)
        
        for (index, sample) in selectedSamples.enumerated() {
            let imageURL = URL(fileURLWithPath: sample.imagePath)
            let destinationURL = outputFolder.appendingPathComponent("sample\(index+1)_gt_\(sample.groundTruth)_pred_\(sample.prediction)." + imageURL.pathExtension)
            try? FileManager.default.copyItem(at: imageURL, to: destinationURL)
        }
    }

    static func generateHTMLReport(metrics: Metrics, confusionMatrixURL: URL, sampleFolderURL: URL, outputHTMLURL: URL) {
        var html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Zero-Shot Classification Report</title>
            <style>
                body { font-family: Arial, sans-serif; padding: 20px; }
                h1 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin-top: 20px; }
                th, td { border: 1px solid #ccc; padding: 8px; text-align: center; }
                img { height: 200px; margin: 10px; }
            </style>
        </head>
        <body>
            <h1>Zero-Shot Classification Report</h1>
            <h2>Metrics</h2>
            <ul>
                <li><b>Accuracy:</b> \(String(format: "%.4f", metrics.accuracy))</li>
                <li><b>Precision:</b> \(String(format: "%.4f", metrics.precision))</li>
                <li><b>Recall:</b> \(String(format: "%.4f", metrics.recall))</li>
                <li><b>F1 Score:</b> \(String(format: "%.4f", metrics.f1Score))</li>
            </ul>
            <h2>Confusion Matrix</h2>
            <p><a href="\(confusionMatrixURL.lastPathComponent)">Download CSV</a></p>
            <h2>Sample Predictions</h2>
        """

        let fileManager = FileManager.default
        if let samples = try? fileManager.contentsOfDirectory(at: sampleFolderURL, includingPropertiesForKeys: nil) {
            for imageFile in samples {
                html += "<img src=\"\(sampleFolderURL.lastPathComponent)/\(imageFile.lastPathComponent)\" alt=\"Sample\">"
            }
        }

        html += """
        </body>
        </html>
        """

        try? html.write(to: outputHTMLURL, atomically: true, encoding: .utf8)
    }
}
