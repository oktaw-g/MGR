//
//  ContentView.swift
//  CoreML_Classifier
//
//  Created by Oktawian Głowacz on 18/04/2025.
//
import SwiftUI
import CoreML

struct ContentView: View {
    @State private var selectedModelURL: URL?
    @State private var selectedDatasetURL: URL?
    @State private var selectedOutputFolderURL: URL?
    
    @State private var batchSize: Int = 8

    var body: some View {
        VStack(spacing: 20) {
            // Section: Select paths
            Group {
                Button("Select Model") {
                    selectModel()
                }
                if let model = selectedModelURL {
                    Text("Selected Model: \(model.lastPathComponent)")
                        .font(.subheadline)
                        .foregroundColor(.gray)
                }
                
                Button("Select Dataset Folder") {
                    selectDataset()
                }
                if let dataset = selectedDatasetURL {
                    Text("Selected Dataset: \(dataset.lastPathComponent)")
                        .font(.subheadline)
                        .foregroundColor(.gray)
                }
                
                Button("Select Output Folder") {
                    selectOutputFolder()
                }
                if let output = selectedOutputFolderURL {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Selected Output:")
                            .font(.subheadline)
                            .foregroundColor(.gray)
                        Text(output.path)
                            .font(.system(size: 12, design: .monospaced))
                            .foregroundColor(.primary)
                            .textSelection(.enabled)
                    }
                }

            }
            
            .padding()

            Divider()

            // Section: Run actions
            Group {
                Button("▶️ Run Zero-Shot Evaluation") {
                    runZeroShot()
                }
                .buttonStyle(.borderedProminent)
                .disabled(!allSelectionsMade)

                Button("▶️ Train Model") {
                    runTraining()
                }
                .buttonStyle(.borderedProminent)
                .disabled(!allSelectionsMade)

                Button("▶️ Run Zero-Shot + Train") {
                    runZeroShotAndTrain()
                }
                .buttonStyle(.borderedProminent)
                .disabled(!allSelectionsMade)
            }
            .padding()

            Divider()

            // Section: Batch size setting
            Stepper("Batch Size: \(batchSize)", value: $batchSize, in: 1...64)
                .padding()
        }
        .padding()
    }

    // MARK: - File Pickers
    func selectModel() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.folder, .item]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = true
        panel.canChooseFiles = true
        panel.title = "Select a CoreML Model (.mlmodelc or .mlpackage)"
        
        if panel.runModal() == .OK {
            selectedModelURL = panel.url
        }
    }

    func selectDataset() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.title = "Select Dataset Folder"
        
        if panel.runModal() == .OK {
            selectedDatasetURL = panel.url
        }
    }

    func selectOutputFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.title = "Select Output Folder"
        
        if panel.runModal() == .OK {
            selectedOutputFolderURL = panel.url
        }
    }

    // MARK: - Running Actions

    var allSelectionsMade: Bool {
        selectedModelURL != nil && selectedDatasetURL != nil && selectedOutputFolderURL != nil
    }

    func runZeroShot() {
        guard let modelURL = selectedModelURL,
              let datasetURL = selectedDatasetURL,
              let outputURL = selectedOutputFolderURL else {
            print("❌ Missing selections")
            return
        }
        ZeroShotClassifier.evaluateWithSavePanel(
            modelURL: modelURL,
            datasetURL: datasetURL
        )

    }

    func runTraining() {
        guard let modelURL = selectedModelURL,
              let datasetURL = selectedDatasetURL,
              let outputURL = selectedOutputFolderURL else {
            print("❌ Missing selections")
            return
        }
        Trainer.train(baseModelURL: modelURL, datasetURL: datasetURL, outputFolderURL: outputURL, batchSize: batchSize)
    }

    func runZeroShotAndTrain() {
        runZeroShot()
        runTraining()
    }
}

