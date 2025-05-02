import Foundation
import CoreML
import CreateML
import Vision
import AppKit
import UniformTypeIdentifiers

struct Trainer {
    static func train(baseModelURL: URL, datasetURL: URL, outputFolderURL: URL, batchSize: Int = 16) {
            do {
                let fileManager = FileManager.default
                
                // Create temp folders
                let tempDir = fileManager.temporaryDirectory
                let trainFolder = tempDir.appendingPathComponent("train_data")
                let valFolder = tempDir.appendingPathComponent("val_data")
                let testFolder = tempDir.appendingPathComponent("test_data")
                
                try? fileManager.removeItem(at: trainFolder)
                try? fileManager.removeItem(at: valFolder)
                try? fileManager.removeItem(at: testFolder)
                
                try fileManager.createDirectory(at: trainFolder, withIntermediateDirectories: true)
                try fileManager.createDirectory(at: valFolder, withIntermediateDirectories: true)
                try fileManager.createDirectory(at: testFolder, withIntermediateDirectories: true)
                
                splitDataset(from: datasetURL, trainFolder: trainFolder, valFolder: valFolder, testFolder: testFolder)
                print("[📂] Dataset split into train/val/test folders.")
                
                let trainData = try MLImageClassifier.DataSource.labeledDirectories(at: trainFolder)
                let valData = try MLImageClassifier.DataSource.labeledDirectories(at: valFolder)
                
                let parameters = MLImageClassifier.ModelParameters(
                    validationData: valData,
                    maxIterations: 20,
                    augmentationOptions: [.flip, .exposure, .blur]
                )
                
                let model = try MLImageClassifier(trainingData: trainData, parameters: parameters)
                print("[⚙️] Training complete.")
                
                saveModelWithDialog(model: model, baseModelURL: baseModelURL)

                // Skip evaluation for now if needed
                try? fileManager.removeItem(at: trainFolder)
                try? fileManager.removeItem(at: valFolder)
                try? fileManager.removeItem(at: testFolder)

                print("[✅] Training complete and files cleaned up.")
            } catch {
                print("❌ Trainer error: \(error.localizedDescription)")
            }
        }

    
    // MARK: - Dataset Splitting
    
    static func splitDataset(from sourceFolder: URL, trainFolder: URL, valFolder: URL, testFolder: URL) {
        let fileManager = FileManager.default
        do {
            let classFolders = try fileManager.contentsOfDirectory(at: sourceFolder, includingPropertiesForKeys: nil, options: .skipsHiddenFiles)
            
            for classFolder in classFolders {
                guard classFolder.hasDirectoryPath else { continue }
                
                let images = try fileManager.contentsOfDirectory(at: classFolder, includingPropertiesForKeys: nil, options: .skipsHiddenFiles)
                let shuffled = images.shuffled()
                
                let total = shuffled.count
                let trainCount = Int(Double(total) * 0.6)
                let valCount = Int(Double(total) * 0.2)
                
                let trainImages = shuffled[0..<trainCount]
                let valImages = shuffled[trainCount..<(trainCount+valCount)]
                let testImages = shuffled[(trainCount+valCount)...]
                
                let className = classFolder.lastPathComponent
                
                let trainClassFolder = trainFolder.appendingPathComponent(className)
                let valClassFolder = valFolder.appendingPathComponent(className)
                let testClassFolder = testFolder.appendingPathComponent(className)
                
                try fileManager.createDirectory(at: trainClassFolder, withIntermediateDirectories: true)
                try fileManager.createDirectory(at: valClassFolder, withIntermediateDirectories: true)
                try fileManager.createDirectory(at: testClassFolder, withIntermediateDirectories: true)
                
                for img in trainImages {
                    let dest = trainClassFolder.appendingPathComponent(img.lastPathComponent)
                    try? fileManager.copyItem(at: img, to: dest)
                }
                for img in valImages {
                    let dest = valClassFolder.appendingPathComponent(img.lastPathComponent)
                    try? fileManager.copyItem(at: img, to: dest)
                }
                for img in testImages {
                    let dest = testClassFolder.appendingPathComponent(img.lastPathComponent)
                    try? fileManager.copyItem(at: img, to: dest)
                }
            }
        } catch {
            print("❌ Error splitting dataset: \(error.localizedDescription)")
        }
    }
    
    // MARK: - Evaluation (Manual)
    
    struct Metrics {
        var accuracy: Double
        var precision: Double
        var recall: Double
        var f1Score: Double
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
                       let ciImage = CIImage(data: nsImage.tiffRepresentation!) {
                        
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
            print("❌ Error during manual evaluation: \(error.localizedDescription)")
        }
        
        return (computeMetrics(groundTruths: groundTruths, predictions: predictions))
    }
    
    static func computeMetrics(groundTruths: [String], predictions: [String]) -> (Metrics, [[Int]], [String]) {
        let labels = Array(Set(groundTruths + predictions)).sorted()
        var labelToIndex = [String: Int]()
        for (i, label) in labels.enumerated() {
            labelToIndex[label] = i
        }
        
        var confusion = Array(repeating: Array(repeating: 0, count: labels.count), count: labels.count)
        
        for (gt, pred) in zip(groundTruths, predictions) {
            if let gtIdx = labelToIndex[gt], let predIdx = labelToIndex[pred] {
                confusion[gtIdx][predIdx] += 1
            }
        }
        
        var correct = 0
        for i in 0..<labels.count {
            correct += confusion[i][i]
        }
        let total = groundTruths.count
        let accuracy = Double(correct) / Double(total)
        
        var precisionSum = 0.0
        var recallSum = 0.0
        var f1Sum = 0.0
        
        for i in 0..<labels.count {
            let tp = Double(confusion[i][i])
            let fp = Double((0..<labels.count).map { confusion[$0][i] }.reduce(0, +)) - tp
            let fn = Double((0..<labels.count).map { confusion[i][$0] }.reduce(0, +)) - tp
            
            let precision = tp / max(tp + fp, 1)
            let recall = tp / max(tp + fn, 1)
            let f1 = (2 * precision * recall) / max(precision + recall, 1e-6)
            
            precisionSum += precision
            recallSum += recall
            f1Sum += f1
        }
        
        let avgPrecision = precisionSum / Double(labels.count)
        let avgRecall = recallSum / Double(labels.count)
        let avgF1 = f1Sum / Double(labels.count)
        
        return (Metrics(accuracy: accuracy, precision: avgPrecision, recall: avgRecall, f1Score: avgF1), confusion, labels)
    }
    
    // MARK: - Saving Outputs
    
    static func saveModelWithDialog(model: MLImageClassifier, baseModelURL: URL) {
        let baseModelName = baseModelURL.deletingPathExtension().lastPathComponent

        let panel = NSSavePanel()
        panel.title = "Save Trained Model"
        if let modelUTType = UTType(filenameExtension: "mlmodel") {
            panel.allowedContentTypes = [modelUTType]
        }
        panel.nameFieldStringValue = "TrainedClassifier_\(baseModelName).mlmodel"

        if panel.runModal() == .OK, let saveURL = panel.url {
            do {
                print("📁 Saving model to: \(saveURL.path)")
                if FileManager.default.fileExists(atPath: saveURL.path) {
                    try FileManager.default.removeItem(at: saveURL)
                    print("🧹 Removed existing file.")
                }

                try model.write(to: saveURL)
                print("✅ Model successfully saved to: \(saveURL.path)")

                // 💡 Evaluate model and save metrics
                let outputFolder = saveURL.deletingLastPathComponent()

                let compiledModelURL = try MLModel.compileModel(at: saveURL)
                let mlmodel = try MLModel(contentsOf: compiledModelURL)

                // TEMP folder for test set
                let testFolder = FileManager.default.temporaryDirectory.appendingPathComponent("test_data")

                let (metrics, matrix, labels) = evaluateModel(model: mlmodel, testFolder: testFolder)
                saveConfusionMatrix(matrix: matrix, classLabels: labels, outputURL: outputFolder.appendingPathComponent("confusion_matrix.csv"))
                generateHTMLReport(metrics: metrics, outputFolderURL: outputFolder)

            } catch {
                print("❌ Failed to save or evaluate model: \(error.localizedDescription)")
            }
        } else {
            print("❌ User cancelled save panel.")
        }
    }





    
    static func saveConfusionMatrix(matrix: [[Int]], classLabels: [String], outputURL: URL) {
        var csv = "GroundTruth/Predicted," + classLabels.joined(separator: ",") + "\n"
        for (i, row) in matrix.enumerated() {
            csv += classLabels[i] + "," + row.map { String($0) }.joined(separator: ",") + "\n"
        }
        try? csv.write(to: outputURL, atomically: true, encoding: .utf8)
    }
    
    static func generateHTMLReport(metrics: Metrics, outputFolderURL: URL) {
        let htmlContent = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Training Report</title>
            <style>
                body { font-family: Arial, sans-serif; padding: 20px; }
                h1 { color: #333; }
                ul { list-style-type: none; }
            </style>
        </head>
        <body>
            <h1>Training Report</h1>
            <ul>
                <li><b>Accuracy:</b> \(String(format: "%.4f", metrics.accuracy))</li>
                <li><b>Precision:</b> \(String(format: "%.4f", metrics.precision))</li>
                <li><b>Recall:</b> \(String(format: "%.4f", metrics.recall))</li>
                <li><b>F1 Score:</b> \(String(format: "%.4f", metrics.f1Score))</li>
            </ul>
            <p>See <b>confusion_matrix.csv</b> for confusion matrix details.</p>
        </body>
        </html>
        """
        let htmlURL = outputFolderURL.appendingPathComponent("training_report.html")
        try? htmlContent.write(to: htmlURL, atomically: true, encoding: .utf8)
    }
}
